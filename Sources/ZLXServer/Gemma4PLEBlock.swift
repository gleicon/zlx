import MLX
import MLXNN

/// Gemma 4 Per-Layer Embeddings (PLE) Block
/// 
/// Implements the PLE architecture for Gemma 4 E4B:
/// - Context-aware gating with GELU activation
/// - Token-identity combination via elementwise multiply
/// - Projection back to hidden size
/// - RMSNorm for final output
public class Gemma4PLEBlock: Module {
    let inputGate: Linear
    let projection: Linear
    let norm: RMSNorm
    let pleDim: Int
    
    public init(hiddenSize: Int, pleDim: Int = 256) {
        self.pleDim = pleDim
        self.inputGate = Linear(inputDimensions: hiddenSize, outputDimensions: pleDim, bias: false)
        self.projection = Linear(inputDimensions: pleDim, outputDimensions: hiddenSize, bias: false)
        self.norm = RMSNorm(dimensions: hiddenSize)
        super.init()
    }
    
    /// Forward pass for PLE block
    /// - Parameters:
    ///   - x: Current hidden state [batch, seqLen, hiddenSize]
    ///   - pleSlice: PLE for this layer [batch, seqLen, pleDim]
    /// - Returns: PLE contribution to add to residual [batch, seqLen, hiddenSize]
    public func callAsFunction(_ x: MLXArray, pleSlice: MLXArray) -> MLXArray {
        // 1. Context-aware gate with GELU
        let gate = gelu(inputGate(x))
        
        // 2. Combine with token PLE: gate * pleSlice
        let gated = gate * pleSlice
        
        // 3. Project back to hidden size and normalize
        let pleOut = norm(projection(gated))
        
        // 4. Return residual addition (caller adds to x)
        return pleOut
    }
}
