using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using SaiAgent001;
class Program
{
    static double[] D(JsonElement e,string key)=>e.GetProperty(key).EnumerateArray().Select(x=>x.GetDouble()).ToArray();
    static float[] F(JsonElement e,string key)=>D(e,key).Select(x=>(float)x).ToArray();
    static int Main(string[] args)
    {
        using var doc=JsonDocument.Parse(File.ReadAllText(args[0]));double maximum=0;int n=0;
        foreach(var row in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            var cmd=D(row,"command");double crouch=row.GetProperty("crouch").GetDouble();
            var obs=SaiContract.Observe(D(row,"q"),D(row,"v"),cmd[0],cmd[1],crouch,F(row,"previous"),row.GetProperty("time").GetDouble(),D(row,"heights"));
            var target=SaiContract.Targets(F(row,"action"),cmd[0],cmd[1],crouch);
            var expected=D(row,"observation");var expectedTarget=D(row,"target");
            for(int i=0;i<82;i++)maximum=Math.Max(maximum,Math.Abs(obs[i]-expected[i]));
            for(int i=0;i<16;i++)maximum=Math.Max(maximum,Math.Abs(target[i]-expectedTarget[i]));
            n++;
        }
        Console.WriteLine(JsonSerializer.Serialize(new{cases=n,maximum_absolute_error=maximum,passed=maximum<1e-5}));
        return maximum<1e-5?0:1;
    }
}
