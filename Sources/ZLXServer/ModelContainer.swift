import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Container for a loaded model with generation capabilities
public class ModelContainer {
    public let info: ModelInfo
    public let model: any LanguageModel
    public let tokenizer: Tokenizer
    public let configuration: ModelConfiguration
    
    private var kvCache: KVCache?
    private let maxKVSize: Int
    
    public init(
        info: ModelInfo,
        model: any LanguageModel,
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
        repetitionPenalty: Float = 1.0,
        stream: Bool = false
    ) async throws -> GenerationResult {
        // Apply chat template
        let prompt = try applyChatTemplate(messages: messages)
        
        // Tokenize
        let inputTokens = try tokenizer.encode(text: prompt, addSpecialTokens: true)
        
        // Create or reset KV cache
        if kvCache == nil || kvCache!.size > maxKVSize {
            kvCache = model.createKVCache()
        }
        
        // Generate
        var outputTokens: [Int] = []
        var currentTokens = inputTokens
        
        for _ in 0..<maxTokens {
            // Forward pass
            let logits = model(currentTokens, cache: kvCache)
            
            // Sample next token
            let nextToken = sample(
                logits: logits,
                temperature: temperature,
                topP: topP,
                topK: topK,
                repetitionPenalty: repetitionPenalty,
                previousTokens: outputTokens
            )
            
            // Check for EOS
            if isEOSToken(nextToken) {
                break
            }
            
            outputTokens.append(nextToken)
            currentTokens = [nextToken]
        }
        
        // Decode output
        let outputText = tokenizer.decode(tokens: outputTokens, skipSpecialTokens: true)
        
        return GenerationResult(
            text: outputText,
            promptTokens: inputTokens.count,
            completionTokens: outputTokens.count,
            finishReason: outputTokens.count >= maxTokens ? "length" : "stop"
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
        AsyncStream { continuation in
            Task {
                do {
                    let prompt = try applyChatTemplate(messages: messages)
                    let inputTokens = try tokenizer.encode(text: prompt, addSpecialTokens: true)
                    
                    if kvCache == nil {
                        kvCache = model.createKVCache()
                    }
                    
                    var currentTokens = inputTokens
                    
                    for i in 0..<maxTokens {
                        let logits = model(currentTokens, cache: kvCache)
                        let nextToken = sample(
                            logits: logits,
                            temperature: temperature,
                            topP: topP,
                            topK: topK,
                            repetitionPenalty: 1.0,
                            previousTokens: []
                        )
                        
                        if isEOSToken(nextToken) {
                            continuation.yield(StreamChunk(
                                token: "",
                                isLast: true,
                                index: i
                            ))
                            break
                        }
                        
                        let tokenText = tokenizer.decode(tokens: [nextToken], skipSpecialTokens: false)
                        continuation.yield(StreamChunk(
                            token: tokenText,
                            isLast: false,
                            index: i
                        ))
                        
                        currentTokens = [nextToken]
                    }
                    
                    continuation.finish()
                } catch {
                    continuation.finish()
                }
            }
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
    
    private func sample(
        logits: MLXArray,
        temperature: Float,
        topP: Float,
        topK: Int,
        repetitionPenalty: Float,
        previousTokens: [Int]
    ) -> Int {
        var adjustedLogits = logits
        
        // Apply temperature
        if temperature != 1.0 {
            adjustedLogits = adjustedLogits / temperature
        }
        
        // Apply repetition penalty
        if repetitionPenalty != 1.0 && !previousTokens.isEmpty {
            for token in Set(previousTokens) {
                // Penalize repeated tokens
                // This is simplified - full implementation would modify logits
            }
        }
        
        // Top-k filtering
        if topK > 0 {
            let topKValues = topK(adjustedLogits, k: topK)
            adjustedLogits = adjustedLogits * topKValues
        }
        
        // Top-p (nucleus) sampling
        if topP < 1.0 {
            let sortedLogits = sort(adjustedLogits, axis: -1)
            let probs = softmax(sortedLogits, axis: -1)
            let cumsumProbs = cumsum(probs, axis: -1)
            // Mask tokens beyond top-p threshold
            // Simplified implementation
        }
        
        // Sample from distribution
        let probs = softmax(adjustedLogits, axis: -1)
        return Int(randomCategorical(probs).item(Int.self))
    }
    
    private func isEOSToken(_ token: Int) -> Bool {
        // Check against model's EOS tokens
        let eosTokens: [Int] = [tokenizer.eosTokenId]
        return eosTokens.contains(token)
    }
}

/// Chat message for conversation
public struct ChatMessage: Codable {
    public let role: String
    public let content: String
    
    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

/// Generation result
public struct GenerationResult {
    public let text: String
    public let promptTokens: Int
    public let completionTokens: Int
    public let finishReason: String
}

/// Streaming chunk
public struct StreamChunk {
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
                // DeepSeek doesn't use explicit system tags in the same way
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
