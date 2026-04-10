import Foundation
import MLX
import MLXLMCommon

/// Stub tokenizer for compilation
public struct SimpleTokenizer: Tokenizer {
    public let eosTokenId: Int = 2
    
    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        // Stub - would use real tokenizer
        return [1, 2, 3] 
    }
    
    public func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        // Stub - would use real tokenizer
        return "Hello from model"
    }
}

/// Stub model for compilation  
public struct SimpleLanguageModel {
    public func callAsFunction(_ tokens: [Int], cache: Any?) -> MLXArray {
        // Stub - would run actual inference
        return MLXArray([0.0])
    }
}

/// Protocol for tokenizer
public protocol Tokenizer: Sendable {
    var eosTokenId: Int { get }
    func encode(text: String, addSpecialTokens: Bool) -> [Int]
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String
}

/// Container for a loaded model with generation capabilities
public class ModelContainer: @unchecked Sendable {
    public let info: ModelInfo
    public let model: SimpleLanguageModel
    public let tokenizer: Tokenizer
    public let configuration: ModelConfiguration
    
    private var kvCache: [Int]?
    private let maxKVSize: Int
    
    public init(
        info: ModelInfo,
        model: SimpleLanguageModel,
        tokenizer: Tokenizer,
        configuration: ModelConfiguration,
        maxKVSize: Int = 4096
    ) {
        self.info = info
        self.model = model
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.maxKVSize = maxKVSize
    }
    
    /// Generate text from messages
    public func generate(
        messages: [ChatMessage],
        maxTokens: Int = 1024,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        topK: Int = 40,
        repetitionPenalty: Float = 1.0
    ) async throws -> GenerationResult {
        // Apply chat template
        let prompt = try applyChatTemplate(messages: messages)
        
        // Tokenize
        let inputTokens = tokenizer.encode(text: prompt, addSpecialTokens: true)
        
        // Stub generation - would call actual MLX model
        var outputTokens: [Int] = []
        
        // Simulate generation
        for i in 0..<min(maxTokens, 10) { // Limit to 10 for stub
            outputTokens.append(i + 100)
            if i > 5 { break } // Simulate early stop
        }
        
        // Decode output
        let outputText = tokenizer.decode(tokens: outputTokens, skipSpecialTokens: true)
        
        return GenerationResult(
            text: outputText,
            promptTokens: inputTokens.count,
            completionTokens: outputTokens.count,
            finishReason: "stop"
        )
    }
    
    /// Generate with streaming
    public func generateStreaming(
        messages: [ChatMessage],
        maxTokens: Int = 1024,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        topK: Int = 40
    ) -> AsyncStream<StreamChunk> {
        // Capture values to avoid self access in @Sendable closure
        let tokenizer = self.tokenizer
        let info = self.info
        
        return AsyncStream { @Sendable continuation in
            Task {
                let prompt = applyChatTemplateSync(messages: messages, architecture: info.architecture)
                let inputTokens = tokenizer.encode(text: prompt, addSpecialTokens: true)
                
                for i in 0..<min(maxTokens, 10) {
                    let tokenText = "Token \(i) "
                    continuation.yield(StreamChunk(
                        token: tokenText,
                        isLast: i >= 5,
                        index: i
                    ))
                    
                    if i >= 5 {
                        break
                    }
                }
                
                continuation.finish()
            }
        }
    }
    
    // Synchronous version for streaming
    private func applyChatTemplateSync(messages: [ChatMessage], architecture: ModelArchitecture) -> String {
        switch architecture {
        case .qwen2_5:
            return QwenChatTemplate.apply(messages: messages)
        case .deepseekCoderV2:
            return DeepSeekChatTemplate.apply(messages: messages)
        case .gemma4:
            return Gemma4ChatTemplate.apply(messages: messages)
        }
    }
    
    // MARK: - Private Helpers
    
    private func applyChatTemplate(messages: [ChatMessage]) throws -> String {
        // Use model-specific chat template
        switch info.architecture {
        case .qwen2_5:
            return QwenChatTemplate.apply(messages: messages)
        case .deepseekCoderV2:
            return DeepSeekChatTemplate.apply(messages: messages)
        case .gemma4:
            return Gemma4ChatTemplate.apply(messages: messages)
        }
    }
}

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
    public let token: String
    public let isLast: Bool
    public let index: Int
}

// MARK: - Chat Templates

struct QwenChatTemplate {
    static func apply(messages: [ChatMessage]) -> String {
        var result = ""
        for message in messages {
            switch message.role {
            case "system":
                result += "<|im_start|>system\n\(message.content)<|im_end|>\n"
            case "user":
                result += "<|im_start|>user\n\(message.content)<|im_end|>\n"
            case "assistant":
                result += "<|im_start|>assistant\n\(message.content)<|im_end|>\n"
            default:
                result += "<|im_start|>\(message.role)\n\(message.content)<|im_end|>\n"
            }
        }
        result += "<|im_start|>assistant\n"
        return result
    }
}

struct DeepSeekChatTemplate {
    static func apply(messages: [ChatMessage]) -> String {
        var result = ""
        for message in messages {
            switch message.role {
            case "system":
                result += message.content + "\n"
            case "user":
                result += "User: \(message.content)\n"
            case "assistant":
                result += "Assistant: \(message.content)\n"
            default:
                result += "\(message.role): \(message.content)\n"
            }
        }
        result += "Assistant:"
        return result
    }
}

struct Gemma4ChatTemplate {
    static func apply(messages: [ChatMessage]) -> String {
        var result = ""
        for message in messages {
            switch message.role {
            case "system":
                result += "<|turn|>system\n\(message.content)<|turn|>\n"
            case "user":
                result += "<|turn|>user\n\(message.content)<|turn|>\n"
            case "assistant":
                result += "<|turn|>assistant\n\(message.content)<|turn|>\n"
            default:
                result += "<|turn|>\(message.role)\n\(message.content)<|turn|>\n"
            }
        }
        result += "<|turn|>assistant\n"
        return result
    }
}
