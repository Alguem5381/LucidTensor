using LucidTensor.Core;

Console.WriteLine("Adaline\n");

var model = new Adaline();

var optimizer = new Sgd(model.Parameters(), learningRate: 0.1f);

var input = new Tensor([1f, 2f]);
var target = Tensor.Scalar(1f);

optimizer.ZeroGrad();

var prediction = model.Forward(input);
var loss = 0.5f * (prediction - target).Square().Mean();

loss.Backward();

Console.WriteLine("Um passo só");
Console.WriteLine($"previsão = {prediction.Item():F3}; perda = {loss.Item():F3}");
Console.WriteLine($"gradiente dos pesos = {model.Weight.Grad}");
Console.WriteLine($"gradiente do bias = {model.Bias.Grad!.Item():F3}");

optimizer.Step();

Console.WriteLine($"novos pesos = {model.Weight}; bias = {model.Bias.Item():F3}");


TrainingSample[] samples =
[
    new(new([0f, 0f]), Tensor.Scalar(0.5f)),
    new(new([1f, 0f]), Tensor.Scalar(2.5f)),
    new(new([0f, 1f]), Tensor.Scalar(-2.5f)),
    new(new([1f, 1f]), Tensor.Scalar(-0.5f))
];

for (var epoch = 0; epoch < 500; epoch++)
    foreach (var sample in samples)
    {
        // Gradientes se acumulam nas folhas; cada amostra começa com eles limpos.
        optimizer.ZeroGrad();

        var error = model.Forward(sample.Input) - sample.Target;

        (0.5f * error.Square().Mean()).Backward();

        optimizer.Step();
    }

Console.WriteLine("Aprendendo y = 2x₁ - 3x₂ + 0,5:");
Console.WriteLine($"  pesos = {model.Weight}; bias = {model.Bias.Item():F6}");

using (Autograd.NoGrad())
{
    var result = model.Forward(new Tensor([2f, -1f]));

    Console.WriteLine($"  entrada [2, -1]: previsão = {result.Item():F6}; esperado = 7,5");
    Console.WriteLine($"  previsão sem grafo: {result.Creator is null}\n");

    if (MathF.Abs(result.Item() - 7.5f) > 0.001f)
        throw new Exception("O Adaline não convergiu.");
}

// Cada passo cria outro tensor. O mesmo parâmetro participa de todos os passos.
var recurrentWeight = Tensor.Scalar(0.5f, requiresGrad: true);

var hidden = Tensor.Scalar(0f);

// Seria possível usar linq aqui com Aggregate
foreach (var value in new[] { 1f, 2f, 3f })
    hidden = recurrentWeight * hidden + value;

var sequenceLoss = 0.5f * hidden.Square();

sequenceLoss.Backward();

var recurrentGradient = recurrentWeight.Grad!.Item();

Console.WriteLine(
    $"BPTT de três passos: estado final = {hidden.Item():F2}; " +
    $"gradiente do peso = {recurrentGradient:F2} (esperado 12,75).");

if (MathF.Abs(recurrentGradient - 12.75f) > 0.001f)
    throw new Exception("O gradiente da recorrência não corresponde ao cálculo esperado.");

internal sealed record TrainingSample(Tensor Input, Tensor Target);
