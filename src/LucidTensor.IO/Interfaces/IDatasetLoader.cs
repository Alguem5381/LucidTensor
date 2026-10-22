namespace LucidTensor.IO.Interfaces;

public interface IDatasetLoader<T>
{
    
    
    /// <summary>
    /// Le arquivo de forma preguiçosa sem carregar tudo de uma vez, e retorna elementos de
    /// um a um
    /// </summary>
    /// <param name="filePath">Caminho do arquivo dado</param>
    /// <param name="cancellationToken"></param>
    /// <returns></returns>
    IAsyncEnumerable<T> LoadDataAsync(string filePath, CancellationToken cancellationToken = default);
    
    
}