import Foundation
import MLX
@preconcurrency import MLXLMCommon

/// Chat message for conversation
public struct ChatMessage: Codable, Sendable {
    public let role: String
    public let content: String
    
    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

/// Generation result
public struct GenerationResult: Sendable {
    public let text: String
    public let promptTokens: Int
    public let completionTokens: Int
    public let finishReason: String
}

/// Streaming chunk
public struct StreamChunk: Sendable {
    public let text: String
    public let isLast: Bool
    public let index: Int
}

/// Wrapper for MLXLMCommon's ModelContainer
/// Note: Uses @unchecked Sendable because MLXLMCommon.ModelContainer already uses SerialAccessContainer for thread-safety
public final class ZLXModelContainer: @unchecked Sendable {
    public let info: ModelInfo
    private let underlying: MLXLMCommon.ModelContainer
    
    public init(info: ModelInfo, container: MLXLMCommon.ModelContainer) {
        self.info = info
        self.underlying = container
    }
    
    /// Convert our ChatMessage to MLXLMCommon.Chat.Message
    private func convertMessages(_ messages: [ChatMessage]) -> [MLXLMCommon.Chat.Message] {
        return messages.map { msg in
            switch msg.role {
            case "system":
                return MLXLMCommon.Chat.Message.system(msg.content)
            case "user":
                return MLXLMCommon.Chat.Message.user(msg.content)
            case "assistant":
                return MLXLMCommon.Chat.Message.assistant(msg.content)
            default:
                return MLXLMCommon.Chat.Message.user(msg.content)
            }
        }
    }
    
    /// Generate text from messages (non-streaming)
    public func generate(
        messages: [ChatMessage],
        maxTokens: Int = 1024,
        temperature: Float = 0.7,
        topP: Float = 0.9
    ) async throws -> GenerationResult {
        // Convert messages
        let mlxMessages = convertMessages(messages)
        
        // Create UserInput
        let userInput = UserInput(chat: mlxMessages)
        
        // Prepare input (applies chat template, tokenizes)
        let lmInput = try await underlying.prepare(input: userInput)
        
        // Create generation parameters
        let params = GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP
        )
        
        // Generate using ModelContainer's generate method
        let stream = try await underlying.generate(input: lmInput, parameters: params)
        
        // Collect all chunks into results
        var text = ""
        var promptTokens = 0
        var completionTokens = 0
        var stopReason = "stop"
        
        for await generation in stream {
            switch generation {
            case .chunk(let chunk):
                text += chunk
            case .info(let info):
                promptTokens = info.promptTokenCount
                completionTokens = info.generationTokenCount
                stopReason = String(describing: info.stopReason)
            case .toolCall:
                break
            }
        }
        
        return GenerationResult(
            text: text,
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            finishReason: stopReason
        )
    }
    
    /// Generate with streaming - returns the underlying stream directly
    public func generateStreaming(
        messages: [ChatMessage],
        maxTokens: Int = 1024,
        temperature: Float = 0.7,
        topP: Float = 0.9
    ) async throws -> AsyncStream<StreamChunk> {
        let mlxMessages = convertMessages(messages)
        
        let userInput = UserInput(chat: mlxMessages)
        let lmInput = try await underlying.prepare(input: userInput)
        let params = GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP
        )
        
        let stream = try await underlying.generate(input: lmInput, parameters: params)
        
        return AsyncStream { continuation in
            // Use a detached task to avoid capturing issues
            let task = Task {
                var index = 0
                for await generation in stream {
                    switch generation {
                    case .chunk(let text):
                        continuation.yield(StreamChunk(
                            text: text,
                            isLast: false,
                            index: index
                        ))
                        index += 1
                    case .info:
                        continuation.yield(StreamChunk(
                            text: "",
                            isLast: true,
                            index: index
                        ))
                        continuation.finish()
                    case .toolCall:
                        break
                    }
                }
                continuation.finish()
            }
            
            // Cancel the task if the stream is terminated early
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }
}
