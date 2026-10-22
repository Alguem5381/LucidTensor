using System.Runtime.CompilerServices;

using LucidTensor.IO.Interfaces;
using LucidTensor.IO.Models;

namespace LucidTensor.IO.Loaders;

public class CsvDatasetLoader : IDatasetLoader<DatasetSample>
{
    private readonly bool _hasHeader;
    private readonly int _inputCount;

    public CsvDatasetLoader(int inputCount, bool hasHeader = true)
    {
        _inputCount = inputCount;
        _hasHeader = hasHeader;
    }

    public async IAsyncEnumerable<DatasetSample> LoadDataAsync(string filePath,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (!File.Exists(filePath))
        {
            yield break;
        }

        using StreamReader reader = new(filePath);

        if (_hasHeader && !reader.EndOfStream)
        {
            await reader.ReadLineAsync(cancellationToken);
        }

        while (!reader.EndOfStream)
        {
            cancellationToken.ThrowIfCancellationRequested();

            string? line = await reader.ReadLineAsync(cancellationToken);
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            string[] values = line.Split(',');
            if (values.Length != _inputCount)
            {
                continue;
            }


            double[] inputs = new double[_inputCount];
            for (int i = 0; i < _inputCount; i++)
            {
                double.TryParse(values[i], out inputs[i]);
            }

            int targetCount = values.Length / _inputCount;
            double[] targets = new double[targetCount];
            for (int i = 0; i < targetCount; i++)
            {
                double.TryParse(values[_inputCount + i], out targets[i]);
            }

            yield return new DatasetSample { Inputs = inputs, Targets = targets };
        }
    }
}