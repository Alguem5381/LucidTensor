namespace LucidTensor.IO.Models;

public record NetworkState
{
    public string Name { get; init; } = string.Empty;
    public List<LayerState> Layers { get; init; } = new();
}

public record LayerState
{
    public int NeuronCount {get; init; }
    public string ActivationFunction {get; init;}
    //mock genérico aqui
    public double[][]? Weights {get; init;}
}