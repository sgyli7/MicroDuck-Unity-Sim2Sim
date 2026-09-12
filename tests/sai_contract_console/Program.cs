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
            bool stairs=row.TryGetProperty("stairs",out var s) && s.GetBoolean();
            double time=row.GetProperty("time").GetDouble();var heights=D(row,"heights");
            var obs=SaiContract.Observe(D(row,"q"),D(row,"v"),cmd[0],cmd[1],crouch,F(row,"previous"),time,heights,stairs?3.2:2.4);
            double lift=row.TryGetProperty("lift_height",out var lh)?lh.GetDouble():.055;
            double scale=row.TryGetProperty("leg_scale",out var ls)?ls.GetDouble():.18;
            var target=stairs?SaiContract.StairTargets(F(row,"action"),cmd[0],cmd[1],crouch,time,heights,lift,scale):SaiContract.Targets(F(row,"action"),cmd[0],cmd[1],crouch);
            var expected=D(row,"observation");var expectedTarget=D(row,"target");
            for(int i=0;i<82;i++)maximum=Math.Max(maximum,Math.Abs(obs[i]-expected[i]));
            for(int i=0;i<16;i++)maximum=Math.Max(maximum,Math.Abs(target[i]-expectedTarget[i]));
            n++;
        }
        if(doc.RootElement.TryGetProperty("heading_cases",out var headingRows))
        {
            var heading=new SaiHeadingHold();
            foreach(var row in headingRows.EnumerateArray())
            {
                var cmd=D(row,"command");var target=D(row,"input_target");
                double limit=row.TryGetProperty("max_correction",out var mc)?mc.GetDouble():.4;
                double? desired=row.TryGetProperty("desired_heading",out var dh) && dh.ValueKind!=JsonValueKind.Null?(double?)dh.GetDouble():null;
                heading.Apply(target,cmd[0],cmd[1],row.GetProperty("yaw").GetDouble(),row.GetProperty("yaw_rate").GetDouble(),limit,desired);
                var expected=D(row,"target");
                for(int i=0;i<16;i++)maximum=Math.Max(maximum,Math.Abs(target[i]-expected[i]));
                n++;
            }
        }
        Console.WriteLine(JsonSerializer.Serialize(new{cases=n,maximum_absolute_error=maximum,passed=maximum<1e-5}));
        return maximum<1e-5?0:1;
    }
}
