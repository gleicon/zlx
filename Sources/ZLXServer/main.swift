import ArgumentParser
import Foundation
import Hummingbird
import MLX
import MLXLMCommon
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
@main
struct ZLXServer: AsyncParsableCommand {
    @Option(name: .shortAndLong, help: "Model path or HuggingFace ID")
    var model: String = "qwen2.5-coder-1.5b"
    
    @Option(name: .shortAndLong, help: "Host to bind to")
    var host: String = "127.0.0.1"
    
    @Option(name: .shortAndLong, help: "Port to listen on")
    var port: Int = 8080
    
    @Option(name: .shortAndLong, help: "Maximum KV cache size")
    var maxKVSize: Int = 4096
    
    @Flag(name: .shortAndLong, help: "Enable verbose logging")
    var verbose: Bool = false
    
    func run() async throws {
        print("🚀 ZLX Server (Swift MLX)")
        print("   Model: \(model)")
        print("   Endpoint: http://\(host):\(port)/v1")
        
        // Initialize MLX
        MLX.GPU.set(cacheLimit: 8 * 1024 * 1024 * 1024) // 8GB GPU cache
        
        // Load model
        let modelContainer: ModelContainer
        do {
            modelContainer = try await ModelLoader.load(
                modelId: model,
                maxKVSize: maxKVSize
            )
            print("✅ Model loaded: \(modelContainer.info.name)")
        } catch {
            print("❌ Failed to load model: \(error)")
            throw error
        }
        
        // Setup HTTP server
        let router = HBRouter()
        
        // Health check
        router.get("/health") { _, _ in
            HBResponse(
                status: .ok,
                body: .init(byteBuffer: ByteBuffer(string: "{\"status\": \"ok\"}"))
            )
        }
        
        // OpenAI-compatible models list
        router.get("/v1/models") { _, _ in
            let registry = await ModelRegistry.shared
            let models = await registry.listModels()
            
            let response = ModelsResponse(
                data: models.map { ModelObject(id: $0.id, object: "model", ownedBy: "local") }
            )
            
            return try HBResponse(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(data: JSONEncoder().encode(response))
            )
        }
        
        // Chat completions endpoint
        router.post("/v1/chat/completions") { request, context in
            let body: ChatCompletionRequest
            do {
                body = try await request.decode(as: ChatCompletionRequest.self, using: JSONDecoder())
            } catch {
                return HBResponse(
                    status: .badRequest,
                    body: .init(string: "{\"error\": \"Invalid JSON\"}")
                )
            }
            
            let targetModelId = body.model ?? self.model
            
            // Get or load model
            let targetModel: ModelContainer
            do {
                targetModel = try await ModelLoader.load(modelId: targetModelId)
            } catch {
                return HBResponse(
                    status: .notFound,
                    body: .init(string: "{\"error\": \"Model not found: \(targetModelId)\"}")
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
        let app = HBApplication(router: router)
        app.configuration.address = .hostname(host, port: port)
        
        print("")
        print("✅ Server ready at http://\(host):\(port)/v1")
        print("   Available models:")
        let registry = await ModelRegistry.shared
        for modelInfo in await registry.listModels() {
            print("     - \(modelInfo.id): \(modelInfo.name)")
        }
        print("")
        
        try await app.run()
    }
    
    // MARK: - Request Handlers
    
    private func handleNonStreaming(
        request: ChatCompletionRequest,
        model: ModelContainer
    ) async throws -> HBResponse {
        let messages = request.messages.map { ChatMessage(role: $0.role, content: $0.content) }
        
        let result = try await model.generate(
            messages: messages,
            maxTokens: request.maxTokens ?? 1024,
            temperature: request.temperature ?? 0.7,
            topP: request.topP ?? 0.9,
            topK: request.topK ?? 40,
            repetitionPenalty: request.repetitionPenalty ?? 1.0
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
        
        return try HBResponse(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(data: JSONEncoder().encode(response))
        )
    }
    
    private func handleStreaming(
        request: ChatCompletionRequest,
        model: ModelContainer,
        context: HBRequestContext
    ) async throws -> HBResponse {
        // Create async stream for SSE
        let stream = AsyncStream<String> { continuation in
            Task {
                let messages = request.messages.map { ChatMessage(role: $0.role, content: $0.content) }
                
                var chunkIndex = 0
                for await chunk in model.generateStreaming(
                    messages: messages,
                    maxTokens: request.maxTokens ?? 1024
                ) {
                    let responseChunk = ChatCompletionChunk(
                        id: "chat-\(UUID().uuidString)",
                        object: "chat.completion.chunk",
                        created: Int(Date().timeIntervalSince1970),
                        model: request.model ?? model.info.id,
                        choices: [
                            ChunkChoice(
                                index: chunkIndex,
                                delta: Delta(content: chunk.token),
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
        for try await chunk in stream {
            body.writeString(chunk)
        }
        
        return HBResponse(
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
