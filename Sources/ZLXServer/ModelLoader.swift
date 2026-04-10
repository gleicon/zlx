import Foundation
import MLX
@preconcurrency import MLXLMCommon

/// Loads models using MLXLMCommon
public struct ModelLoader {

    /// Load a model from HuggingFace ID
    public static func load(
        modelId: String,
        maxKVSize: Int = 4096
    ) async throws -> ZLXModelContainer {
        let registry = ModelRegistry.shared

        // Resolve alias
        let resolvedId = registry.resolveAlias(modelId)

        guard let modelInfo = registry.getModel(id: resolvedId) else {
            throw ModelError.unknownModel(modelId)
        }

        // Check if already loaded
        if registry.isLoaded(id: resolvedId) {
            if let cached = registry.getZLXModel(id: resolvedId) {
                return cached
            }
        }

        print("[ZLX] Loading model: \(modelInfo.name)")

        // Load using MLXLMCommon's global loadModelContainer function
        do {
            let mlxContainer = try await loadModelContainer(id: modelInfo.id)

            let container = ZLXModelContainer(
                info: modelInfo,
                container: mlxContainer
            )

            // Store in registry
            registry.storeLoadedModel(id: resolvedId, model: container)

            print("[ZLX] Model loaded successfully: \(modelInfo.name)")
            return container

        } catch {
            print("[ZLX] Failed to load model: \(error)")
            throw ModelError.failedToLoad(error.localizedDescription)
        }
    }
}

/// Model loading errors
public enum ModelError: Error {
    case unknownModel(String)
    case invalidConfiguration(String)
    case failedToLoad(String)
    case insufficientMemory(required: Double, available: Double)
}
