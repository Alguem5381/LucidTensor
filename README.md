# Protótipo do LucidTensor

Protótipo de diferenciação automática reversa em C# / .NET 10, executado em CPU.

## Executar

```sh
dotnet run --project src/LucidTensor.Console
```

O console mostra um passo de Adaline calculável à mão, aprende
`y = 2x₁ - 3x₂ + 0,5` e executa uma recorrência de três passos com BPTT.
Ele verifica a previsão final `7,5` para `[2, -1]` e o gradiente recorrente `12,75`.

## Organização e limites

- `Tensors/Tensor.cs`: dados, shape, construção e indexação.
- `Tensors/Tensor.Elementwise.cs`: operadores elemento a elemento e broadcasting escalar.
- `Tensors/Tensor.LinearAlgebra.cs`: matrizes, produto escalar e reduções.
- `Tensors/Tensor.Autograd.cs`: registro, atualização controlada e backward.
- `Differentiation/OperationRecord.cs`: dados e regra de derivação de uma execução da operação.
- `Differentiation/Autograd.cs`: escopo de controle do rastreamento.
- `Models/Module.cs`: registro de parâmetros e submódulos.

Não há broadcasting geral, derivadas de ordem superior nem execução concorrente sobre tensores compartilhados.
