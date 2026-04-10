import Foundation
import MLX
import MLXLMCommon

// MARK: - Types

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

// MARK: - Registry

/// Non-actor model registry for synchronous access during initialization
public final class ModelRegistry: @unchecked Sendable {
    public static let shared = ModelRegistry()

    private var models: [String: ModelInfo] = [:]
    private var loadedModels: [String: Any] = [:]
    private let queue = DispatchQueue(label: "zlx.registry", attributes: .concurrent)

    private init() {
        registerDefaultModels()
    }

    /// Register built-in models synchronously
    private func registerDefaultModels() {
        // Qwen 2.5 Coder models
        register(ModelInfo(
            id: "qwen2.5-coder-1.5b",
            name: "Qwen 2.5 Coder 1.5B",
            architecture: .qwen2_5,
            description: "Fast coding model with 1.5B parameters",
            sizeGB: 3.9,
            path: "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
            isLocal: false
        ))

        register(ModelInfo(
            id: "qwen2.5-coder-3b",
            name: "Qwen 2.5 Coder 3B",
            architecture: .qwen2_5,
            description: "Balanced coding model with 3B parameters",
            sizeGB: 6.5,
            path: "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
            isLocal: false
        ))

        register(ModelInfo(
            id: "qwen2.5-coder-7b",
            name: "Qwen 2.5 Coder 7B",
            architecture: .qwen2_5,
            description: "Powerful coding model with 7B parameters",
            sizeGB: 14.2,
            path: "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
            isLocal: false
        ))

        // DeepSeek Coder V2
        register(ModelInfo(
            id: "deepseek-coder-v2-lite",
            name: "DeepSeek Coder V2 Lite",
            architecture: .deepseekCoderV2,
            description: "MoE coding model - efficient 16B active params",
            sizeGB: 8.8,
            path: "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit",
            isLocal: false
        ))

        // Gemma 4
        register(ModelInfo(
            id: "gemma-4-e4b",
            name: "Gemma 4 E4B",
            architecture: .gemma4,
            description: "Google Gemma 4 E4B with PLE architecture",
            sizeGB: 10.3,
            path: "mlx-community/gemma-4-e4b-it-4bit",
            isLocal: false
        ))
    }

    /// Register a model
    public func register(_ model: ModelInfo) {
        queue.async(flags: .barrier) {
            self.models[model.id] = model
        }
    }

    /// Get model info by ID
    public func getModel(id: String) -> ModelInfo? {
        return queue.sync {
            models[id]
        }
    }

    /// List all registered models
    public func listModels() -> [ModelInfo] {
        return queue.sync {
            Array(models.values).sorted { $0.id < $1.id }
        }
    }

    /// Resolve a model alias to full ID
    public func resolveAlias(_ alias: String) -> String {
        // Check if it's a direct match first
        if queue.sync(execute: { models[alias] }) != nil {
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
        return queue.sync {
            loadedModels[id] != nil
        }
    }

    /// Store a loaded model
    public func storeLoadedModel(id: String, model: Any) {
        queue.async(flags: .barrier) {
            self.loadedModels[id] = model
        }
    }

    /// Get a loaded ZLXModelContainer specifically
    public func getZLXModel(id: String) -> ZLXModelContainer? {
        return queue.sync {
            loadedModels[id] as? ZLXModelContainer
        }
    }

    /// Unload a model to free memory
    public func unloadModel(id: String) {
        queue.async(flags: .barrier) {
            self.loadedModels.removeValue(forKey: id)
        }
    }

    /// Get memory estimate for a model
    public func estimateMemoryGB(modelId: String) -> Double? {
        return queue.sync {
            models[modelId]?.sizeGB
        }
    }
}
