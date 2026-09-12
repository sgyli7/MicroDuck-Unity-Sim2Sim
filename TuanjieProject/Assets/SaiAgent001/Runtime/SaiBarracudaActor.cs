using System;
using Unity.Barracuda;

namespace SaiAgent001
{
    public sealed class SaiBarracudaActor : IDisposable
    {
        private readonly IWorker worker;
        private readonly string inputName,outputName;
        public SaiBarracudaActor(NNModel policy)
        {
            var net=ModelLoader.Load(policy);
            if(net.inputs.Count!=1 || net.outputs.Count!=1)throw new InvalidOperationException("Sai ONNX needs one input/output");
            inputName=net.inputs[0].name;outputName=net.outputs[0];
            worker=WorkerFactory.CreateWorker(WorkerFactory.Type.CSharpBurst,net);
        }
        public float[] Infer(float[] observation)
        {
            using(var input=new Tensor(1,82,observation,inputName))
            {
                worker.Execute(input);var output=worker.PeekOutput(outputName);
                if(output.length!=16)throw new InvalidOperationException("Sai ONNX output is not 16D");
                var result=new float[16];for(int i=0;i<16;i++)result[i]=output[i];return result;
            }
        }
        public void Dispose(){worker.Dispose();}
    }
}
