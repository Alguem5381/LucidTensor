using System.Diagnostics;
using Xunit;

using LucidTensor.IO.Models;
using LucidTensor.IO.Serializers;


namespace LucidTensor.Tests.IoTests;

    

public class JsonSerializerTeste
{

    [Fact]    
    public async Task IsValidAsyncTest_true()
    {
        var serializer = new JsonNetworkSerializer();
        var filePath = "json_valido.json";
        var network = new NetworkState
        {
            Name = "Rede valida", Layers = new List<LayerState> { new LayerState { NeuronCount = 5 } }
        };
        await serializer.SaveAsync(network, filePath);

        var isValid = await serializer.IsValidAsync(filePath);
        
        Assert.True(isValid);
        
        if(File.Exists(filePath)) File.Delete(filePath);
    }

    [Fact]
    public async Task IsValidAsyncTest_false()
    {
        var serializer = new JsonNetworkSerializer();
        var filePath = "json_valido.json";
        //grava um json inválido
        await File.WriteAllTextAsync(filePath, "{json-invalido}");
        //isValidAsync lança uma exceção e retorna false
        var isValid = await serializer.IsValidAsync(filePath);
        Assert.False(isValid);
        if(File.Exists(filePath)) File.Delete(filePath);
    }
    
}