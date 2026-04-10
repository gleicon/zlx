import ArgumentParser
import Foundation
import Hummingbird
import MLX
import MLXLMCommon
import MLXLLM
import NIOCore

/// ZLX Server - OpenAI-compatible inference server in pure Swift
///
/// Provides:
/// - Qwen 2.5 Coder support
/// - DeepSeek Coder V2 support
/// - Gemma 4 E4B support with PLE (Per-Layer Embeddings)
/// - Native MLX inference on Apple Silicon
/// - OpenAI-compatible /v1/chat/completions endpoint
/// - Streaming and non-streaming responses
@available(macOS 10.15, iOS 13, tvOS 13, watchOS 6, *)
struct ZLXServer: AsyncParsableCommand {
    @Option(name: .shortAndLong, help: "Model path or HuggingFace ID")
    var model: String = "qwen2.5-coder-1.5b"

    @Option(name: .shortAndLong, help: "Host to bind to")
    var host: String = "127.0.0.1"

    @Option(name: .shortAndLong, help: "Port to listen on")
    var port: Int = 8080

    @Option(name: [.long, .customShort("k")], help: "Maximum KV cache size")
    var maxKVSize: Int = 4096

    @Flag(name: .shortAndLong, help: "Enable verbose logging")
    var verbose: Bool = false

    func run() async throws {
        // Initialize LLM model factories
        // This registers the LLM models with MLXLMCommon's ModelFactoryRegistry
        _ = LLMModelFactory.shared

        print("[ZLX] Starting ZLX Server (Swift MLX)")
        print("[ZLX] Model: \(model)")
        print("[ZLX] Endpoint: http://\(host):\(port)/v1")

        // Initialize MLX
        MLX.GPU.set(cacheLimit: 8 * 1024 * 1024 * 1024) // 8GB GPU cache

        // Load model
        let modelContainer: ZLXModelContainer
        do {
            modelContainer = try await ModelLoader.load(
                modelId: model,
                maxKVSize: maxKVSize
            )
            print("[ZLX] Model loaded: \(modelContainer.info.name)")
        } catch {
            print("[ZLX] Failed to load model: \(error)")
            throw error
        }

        // Setup HTTP server
        let router = Router()

        // Health check
        router.get("/health") { _, _ -> Response in
            Response(
                status: .ok,
                body: .init(byteBuffer: ByteBuffer(string: "{\"status\": \"ok\"}"))
            )
        }

        // OpenAI-compatible models list
        router.get("/v1/models") { _, _ -> Response in
            let registry = ModelRegistry.shared
            let models = registry.listModels()

            let response = ModelsResponse(
                data: models.map { ModelObject(id: $0.id, object: "model") }
            )

            do {
                let data = try JSONEncoder().encode(response)
                let buffer = ByteBuffer(data: data)
                return Response(
                    status: .ok,
                    headers: [.contentType: "application/json"],
                    body: .init(byteBuffer: buffer)
                )
            } catch {
                let errorBuffer = ByteBuffer(string: "Encoding error")
                return Response(status: .internalServerError, body: .init(byteBuffer: errorBuffer))
            }
        }

        // Chat completions endpoint
        router.post("/v1/chat/completions") { request, context -> Response in
            // Collect request body
            var buffer = ByteBuffer()
            do {
                for try await chunk in request.body {
                    var mutableChunk = chunk
                    buffer.writeBuffer(&mutableChunk)
                }
            } catch {
                let errorBuffer = ByteBuffer(string: "{\"error\": \"Failed to read request body\"}")
                return Response(
                    status: .badRequest,
                    body: .init(byteBuffer: errorBuffer)
                )
            }

            let body: ChatCompletionRequest
            do {
                guard let data = buffer.getData(at: 0, length: buffer.readableBytes) else {
                    throw DecodingError.dataCorrupted(.init(codingPath: [], debugDescription: "Empty body"))
                }
                body = try JSONDecoder().decode(ChatCompletionRequest.self, from: data)
            } catch {
                let errorBuffer = ByteBuffer(string: "{\"error\": \"Invalid JSON: \(error)\"}")
                return Response(
                    status: .badRequest,
                    body: .init(byteBuffer: errorBuffer)
                )
            }

            let targetModelId = body.model ?? self.model

            // Get or load model
            let targetModel: ZLXModelContainer
            do {
                targetModel = try await ModelLoader.load(modelId: targetModelId)
            } catch {
                let errorBuffer = ByteBuffer(string: "{\"error\": \"Model not found: \(targetModelId)\"}")
                return Response(
                    status: .notFound,
                    body: .init(byteBuffer: errorBuffer)
                )
            }

            if body.stream == true {
                // Streaming response
                return try await self.handleStreaming(request: body, model: targetModel, context: context)
            } else {
                // Non-streaming response
                return try await self.handleNonStreaming(request: body, model: targetModel)
            }
        }

        // Build and start server
        var app = Application(router: router)
        app.configuration.address = .hostname(host, port: port)

        print("")
        print("[ZLX] Server ready at http://\(host):\(port)/v1")
        print("[ZLX] Available models:")
        let registry = ModelRegistry.shared
        for modelInfo in registry.listModels() {
            print("[ZLX]   - \(modelInfo.id): \(modelInfo.name)")
        }
        print("")

        try await app.run()
    }

    // MARK: - Request Handlers

    private func handleNonStreaming(
        request: ChatCompletionRequest,
        model: ZLXModelContainer
    ) async throws -> Response {
        let messages = request.messages.map { ChatMessage(role: $0.role, content: $0.content) }

        let result = try await model.generate(
            messages: messages,
            maxTokens: request.maxTokens ?? 1024,
            temperature: request.temperature ?? 0.7,
            topP: request.topP ?? 0.9
        )

        let response = ChatCompletionResponse(
            id: "chat-\(UUID().uuidString)",
            object: "chat.completion",
            created: Int(Date().timeIntervalSince1970),
            model: request.model ?? model.info.id,
            choices: [
                Choice(
                    index: 0,
                    message: Message(role: "assistant", content: result.text),
                    finishReason: result.finishReason
                )
            ],
            usage: Usage(
                promptTokens: result.promptTokens,
                completionTokens: result.completionTokens,
                totalTokens: result.promptTokens + result.completionTokens
            )
        )

        do {
            let data = try JSONEncoder().encode(response)
            let buffer = ByteBuffer(data: data)
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: buffer)
            )
        } catch {
            let errorBuffer = ByteBuffer(string: "Encoding error")
            return Response(status: .internalServerError, body: .init(byteBuffer: errorBuffer))
        }
    }

    private func handleStreaming(
        request: ChatCompletionRequest,
        model: ZLXModelContainer,
        context: any RequestContext
    ) async throws -> Response {
        let messages = request.messages.map { ChatMessage(role: $0.role, content: $0.content) }

        // Get the streaming response from the model
        let stream = try await model.generateStreaming(
            messages: messages,
            maxTokens: request.maxTokens ?? 1024
        )

        // Create async stream for SSE
        let sseStream = AsyncStream<String> { continuation in
            Task {
                var chunkIndex = 0
                for await chunk in stream {
                    let responseChunk = ChatCompletionChunk(
                        id: "chat-\(UUID().uuidString)",
                        object: "chat.completion.chunk",
                        created: Int(Date().timeIntervalSince1970),
                        model: request.model ?? model.info.id,
                        choices: [
                            ChunkChoice(
                                index: chunkIndex,
                                delta: Delta(content: chunk.text),
                                finishReason: chunk.isLast ? "stop" : nil
                            )
                        ]
                    )

                    if let data = try? JSONEncoder().encode(responseChunk),
                       let json = String(data: data, encoding: .utf8) {
                        continuation.yield("data: \(json)\n\n")
                    }

                    if chunk.isLast {
                        continuation.yield("data: [DONE]\n\n")
                        continuation.finish()
                    }

                    chunkIndex += 1
                }
            }
        }

        // Collect stream into response body
        var body = ByteBuffer()
        for try await chunk in sseStream {
            body.writeString(chunk)
        }

        return Response(
            status: .ok,
            headers: [
                .contentType: "text/event-stream",
                .cacheControl: "no-cache",
            ],
            body: .init(byteBuffer: body)
        )
    }
}

// MARK: - Request/Response Types

struct ChatCompletionRequest: Codable {
    let model: String?
    let messages: [RequestMessage]
    let stream: Bool?
    let maxTokens: Int?
    let temperature: Float?
    let topP: Float?
    let topK: Int?
    let repetitionPenalty: Float?

    enum CodingKeys: String, CodingKey {
        case model, messages, stream
        case maxTokens = "max_tokens"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
    }
}

struct RequestMessage: Codable {
    let role: String
    let content: String
}

struct ModelsResponse: Codable {
    let data: [ModelObject]
}

struct ModelObject: Codable {
    let id: String
    let object: String
    let ownedBy: String = "local"

    enum CodingKeys: String, CodingKey {
        case id, object
        case ownedBy = "owned_by"
    }
}

struct ChatCompletionResponse: Codable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [Choice]
    let usage: Usage
}

struct Choice: Codable {
    let index: Int
    let message: Message
    let finishReason: String

    enum CodingKeys: String, CodingKey {
        case index, message
        case finishReason = "finish_reason"
    }
}

struct Message: Codable {
    let role: String
    let content: String
}

struct Usage: Codable {
    let promptTokens: Int
    let completionTokens: Int
    let totalTokens: Int

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }
}

struct ChatCompletionChunk: Codable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [ChunkChoice]
}

struct ChunkChoice: Codable {
    let index: Int
    let delta: Delta
    let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case index, delta
        case finishReason = "finish_reason"
    }
}

struct Delta: Codable {
    let content: String
}

// MARK: - Entry Point

// Run the server
Task {
    if #available(macOS 10.15, iOS 13, tvOS 13, watchOS 6, *) {
        try await ZLXServer.main()
    }
}

// Keep main thread alive
RunLoop.main.run()
