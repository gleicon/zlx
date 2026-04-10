import Foundation
import MLX
import MLXLMCommon

/// Loads models using MLXLMCommon with custom configuration
public struct ModelLoader {
    
    /// Load a model from path or HuggingFace ID
    public static func load(
        modelId: String,
        maxKVSize: Int = 4096
    ) async throws -> ModelContainer {
        let registry = await ModelRegistry.shared
        
        // Resolve alias
        let resolvedId = await registry.resolveAlias(modelId)
        
        guard let modelInfo = await registry.getModel(id: resolvedId) else {
            throw ModelError.unknownModel(modelId)
        }
        
        // Check if already loaded
        if await registry.isLoaded(id: resolvedId),
           let cached = await registry.getLoadedModel(id: resolvedId) as? ModelContainer {
            return cached
        }
        
        print("Loading model: \(modelInfo.name)...")
        
        // Load configuration
        let config = try ModelConfiguration.load(from: modelInfo.path)
        
        // Load using MLXLMCommon
        let modelContainer: ModelContainer
        
        switch modelInfo.architecture {
        case .qwen2_5:
            modelContainer = try await loadQwenModel(
                info: modelInfo,
                config: config,
                maxKVSize: maxKVSize
            )
            
        case .deepseekCoderV2:
            modelContainer = try await loadDeepSeekModel(
                info: modelInfo,
                config: config,
                maxKVSize: maxKVSize
            )
            
        case .gemma4:
            modelContainer = try await loadGemma4Model(
                info: modelInfo,
                config: config,
                maxKVSize: maxKVSize
            )
        }
        
        // Store in registry
        await registry.storeLoadedModel(id: resolvedId, model: modelContainer)
        
        print("✅ Model loaded: \(modelInfo.name)")
        return modelContainer
    }
    
    // MARK: - Model-Specific Loaders
    
    private static func loadQwenModel(
        info: ModelInfo,
        config: ModelConfiguration,
        maxKVSize: Int
    ) async throws -> ModelContainer {
        // Use MLXLMCommon's built-in Qwen support
        let modelContainer = try await LLMModelFactory.shared.loadContainer(
            configuration: Configuration(
                id: info.id,
                modelDirectory: info.isLocal ? info.path : nil
            )
        ) { configuration in
            // Custom Qwen configuration if needed
            return try await LLMModelFactory.shared._load(
                configuration: configuration,
                modelType: Qwen2_5Configuration.self
            )
        }
        
        return ModelContainer(
            info: info,
            model: modelContainer.model,
            tokenizer: modelContainer.tokenizer,
            configuration: config,
            maxKVSize: maxKVSize
        )
    }
    
    private static func loadDeepSeekModel(
        info: ModelInfo,
        config: ModelConfiguration,
        maxKVSize: Int
    ) async throws -> ModelContainer {
        // DeepSeek uses standard transformer but with MoE
        let modelContainer = try await LLMModelFactory.shared.loadContainer(
            configuration: Configuration(id: info.id)
        ) { configuration in
            return try await LLMModelFactory.shared._load(
                configuration: configuration,
                modelType: DeepSeekV2Configuration.self
            )
        }
        
        return ModelContainer(
            info: info,
            model: modelContainer.model,
            tokenizer: modelContainer.tokenizer,
            configuration: config,
            maxKVSize: maxKVSize
        )
    }
    
    private static func loadGemma4Model(
        info: ModelInfo,
        config: ModelConfiguration,
        maxKVSize: Int
    ) async throws -> ModelContainer {
        // Gemma 4 requires custom PLE handling
        // For now, fall back to standard Gemma loading via MLXLMCommon
        // Full PLE implementation would use Gemma4DecoderLayer
        
        let modelContainer = try await LLMModelFactory.shared.loadContainer(
            configuration: Configuration(id: info.id)
        )
        
        return ModelContainer(
            info: info,
            model: modelContainer.model,
            tokenizer: modelContainer.tokenizer,
            configuration: config,
            maxKVSize: maxKVSize
        )
    }
}

/// Model loading errors
public enum ModelError: Error {
    case unknownModel(String)
    case invalidConfiguration(String)
    case failedToLoad(String)
    case insufficientMemory(required: Double, available: Double)
}

// MARK: - Configuration Types

private struct Qwen2_5Configuration: Codable {
    let hiddenSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
    }
}

private struct DeepSeekV2Configuration: Codable {
    let hiddenSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let numExperts: Int?
    let numExpertsPerToken: Int?
    
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
    }
}

// MARK: - MLXLMCommon Extensions

extension LLMModelFactory {
    /// Internal loading method with custom configuration type
    func _load<Config: Codable>(
        configuration: Configuration,
        modelType: Config.Type
    ) async throws -> (model: any LanguageModel, tokenizer: Tokenizer) {
        // This would use MLXLMCommon's internal loading
        // For now, return the standard load
        let container = try await loadContainer(configuration: configuration)
        return (container.model, container.tokenizer)
    }
}
