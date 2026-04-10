import Foundation
import MLX
import MLXLMCommon

/// Supported model architectures
public enum ModelArchitecture: String, CaseIterable, Sendable {
    case qwen2_5 = "qwen2.5"
    case deepseekCoderV2 = "deepseek-coder-v2"
    case gemma4 = "gemma4"
    
    public var displayName: String {
        switch self {
        case .qwen2_5: return "Qwen 2.5"
        case .deepseekCoderV2: return "DeepSeek Coder V2"
        case .gemma4: return "Gemma 4"
        }
    }
}

/// Model configuration from config.json
public struct ModelConfiguration: Codable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let intermediateSize: Int
    public let rmsNormEps: Double
    public let vocabSize: Int
    public let ropeTheta: Double
    public let maxPositionEmbeddings: Int
    public let tieWordEmbeddings: Bool
    public let torchDtype: String
    
    // Optional fields for specific architectures
    public let slidingWindow: Int?           // Gemma
    public let hiddenSizePerLayerInput: Int? // Gemma 4 PLE
    public let numExperts: Int?              // MoE models
    public let numExpertsPerToken: Int?      // MoE models
    
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case torchDtype = "torch_dtype"
        case slidingWindow = "sliding_window"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case numExperts = "num_local_experts"
        case numExpertsPerToken = "num_experts_per_tok"
    }
    
    /// Load configuration from a model directory
    public static func load(from path: String) throws -> ModelConfiguration {
        let configPath = (path as NSString).appendingPathComponent("config.json")
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        let decoder = JSONDecoder()
        
        // Try loading with text_config wrapper (Gemma 4 style)
        if let wrapper = try? decoder.decode(ConfigWrapper.self, from: data) {
            return wrapper.textConfig
        }
        
        // Direct config (Qwen, DeepSeek style)
        return try decoder.decode(ModelConfiguration.self, from: data)
    }
}

/// Wrapper for configs with text_config (Gemma 4)
private struct ConfigWrapper: Codable {
    let textConfig: ModelConfiguration
    
    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }
}

/// Model metadata for registry
public struct ModelInfo: Identifiable, Sendable {
    public let id: String
    public let name: String
    public let architecture: ModelArchitecture
    public let description: String
    public let sizeGB: Double
    public let defaultQuantization: String
    public let path: String
    public let isLocal: Bool
    
    public init(
        id: String,
        name: String,
        architecture: ModelArchitecture,
        description: String,
        sizeGB: Double,
        defaultQuantization: String = "4bit",
        path: String,
        isLocal: Bool = true
    ) {
        self.id = id
        self.name = name
        self.architecture = architecture
        self.description = description
        self.sizeGB = sizeGB
        self.defaultQuantization = defaultQuantization
        self.path = path
        self.isLocal = isLocal
    }
}

/// Model registry - manages available models
public actor ModelRegistry {
    public static let shared: ModelRegistry = {
        let registry = ModelRegistry()
        // Register models in a Task since we're outside the actor
        Task {
            await registry.registerDefaultModels()
        }
        return registry
    }()
    
    private var models: [String: ModelInfo] = [:]
    private var loadedModels: [String: Any] = [:] // Type-erased loaded models
    
    private init() {}
    
    /// Register built-in models
    public func registerDefaultModels() {
        // Qwen 2.5 Coder models
        register(ModelInfo(
            id: "qwen2.5-coder-1.5b",
            name: "Qwen 2.5 Coder 1.5B",
            architecture: .qwen2_5,
            description: "Fast coding model with 1.5B parameters",
            sizeGB: 3.9,
            path: "./models/Qwen2.5-Coder-1.5B-Instruct-4bit",
            isLocal: true
        ))
        
        register(ModelInfo(
            id: "qwen2.5-coder-3b",
            name: "Qwen 2.5 Coder 3B",
            architecture: .qwen2_5,
            description: "Balanced coding model with 3B parameters",
            sizeGB: 6.5,
            path: "./models/Qwen2.5-Coder-3B-Instruct-4bit",
            isLocal: true
        ))
        
        register(ModelInfo(
            id: "qwen2.5-coder-7b",
            name: "Qwen 2.5 Coder 7B",
            architecture: .qwen2_5,
            description: "Powerful coding model with 7B parameters",
            sizeGB: 14.2,
            path: "./models/Qwen2.5-Coder-7B-Instruct-4bit",
            isLocal: true
        ))
        
        // DeepSeek Coder V2
        register(ModelInfo(
            id: "deepseek-coder-v2-lite",
            name: "DeepSeek Coder V2 Lite",
            architecture: .deepseekCoderV2,
            description: "MoE coding model - efficient 16B active params",
            sizeGB: 8.8,
            path: "./models/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
            isLocal: true
        ))
        
        // Gemma 4
        register(ModelInfo(
            id: "gemma-4-e4b",
            name: "Gemma 4 E4B",
            architecture: .gemma4,
            description: "Google Gemma 4 E4B with PLE architecture",
            sizeGB: 10.3,
            path: "./models/gemma4-e4b-fixed",
            isLocal: true
        ))
        
        // HuggingFace models (remote)
        register(ModelInfo(
            id: "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
            name: "Qwen 2.5 Coder 1.5B (HF)",
            architecture: .qwen2_5,
            description: "Qwen 2.5 from HuggingFace",
            sizeGB: 3.9,
            path: "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
            isLocal: false
        ))
    }
    
    /// Register a model
    public func register(_ model: ModelInfo) {
        models[model.id] = model
    }
    
    /// Get model info by ID
    public func getModel(id: String) -> ModelInfo? {
        return models[id]
    }
    
    /// List all registered models
    public func listModels() -> [ModelInfo] {
        return Array(models.values).sorted { $0.id < $1.id }
    }
    
    /// Resolve a model alias to full ID
    public func resolveAlias(_ alias: String) -> String {
        // Check if it's a direct match
        if models[alias] != nil {
            return alias
        }
        
        // Check common aliases
        let aliases: [String: String] = [
            "qwen": "qwen2.5-coder-1.5b",
            "qwen-small": "qwen2.5-coder-1.5b",
            "qwen-medium": "qwen2.5-coder-3b",
            "qwen-large": "qwen2.5-coder-7b",
            "deepseek": "deepseek-coder-v2-lite",
            "gemma4": "gemma-4-e4b",
        ]
        
        return aliases[alias.lowercased()] ?? alias
    }
    
    /// Check if a model is loaded
    public func isLoaded(id: String) -> Bool {
        return loadedModels[id] != nil
    }
    
    /// Store a loaded model
    public func storeLoadedModel(id: String, model: Any) {
        loadedModels[id] = model
    }
    
    /// Get a loaded model (generic)
    public func getLoadedModel(id: String) -> Any? {
        return loadedModels[id]
    }
    
    /// Get a loaded ZLXModelContainer specifically
    public func getZLXModel(id: String) -> ZLXModelContainer? {
        return loadedModels[id] as? ZLXModelContainer
    }
    
    /// Unload a model to free memory
    public func unloadModel(id: String) {
        loadedModels.removeValue(forKey: id)
    }
    
    /// Get memory estimate for a model
    public func estimateMemoryGB(modelId: String) -> Double? {
        guard let model = models[modelId] else { return nil }
        return model.sizeGB
    }
}
