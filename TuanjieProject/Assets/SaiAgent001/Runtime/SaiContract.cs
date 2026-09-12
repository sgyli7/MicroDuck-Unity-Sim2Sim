using System;

namespace SaiAgent001
{
    // Public SI state: X forward, Y left, Z up; quaternion w,x,y,z.
    // The ONNX includes normalization. Wheels are velocity targets.
    public static class SaiContract
    {
        public const int Observations = 82, Actions = 16;
        public const double Dt = .02;
        public static readonly string[] Legs = {"front_left", "front_right", "rear_left", "rear_right"};
        public static readonly string[] Axes = {"haa", "hip", "knee", "wheel"};
        public static readonly string[] Arm = {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"};
        public static double Clamp(double x, double lo, double hi) => Math.Max(lo, Math.Min(hi, x));

        public static float[] Observe(double[] q, double[] v, double vx, double wz,
            double crouch, float[] previous, double time, double[] heights)
        {
            if (q.Length != 23 || v.Length != 22 || previous.Length != 16 || heights.Length != 24)
                throw new ArgumentException("Sai state shape mismatch");
            double w=q[3], x=q[4], y=q[5], z=q[6];
            double[] r={1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)};
            var o=new float[82]; int n=0;
            for(int i=0;i<3;i++)o[n++]=(float)r[6+i];
            for(int i=0;i<3;i++)o[n++]=(float)(r[i]*v[0]+r[3+i]*v[1]+r[6+i]*v[2]);
            for(int i=3;i<6;i++)o[n++]=(float)v[i];
            o[n++]=(float)vx;o[n++]=(float)wz;o[n++]=(float)crouch;
            for(int i=0;i<16;i++)if(i%4!=3)o[n++]=(float)q[7+i];
            for(int i=0;i<16;i++)if(i%4!=3)o[n++]=(float)(v[6+i]*.1);
            for(int i=0;i<4;i++)o[n++]=(float)(v[9+4*i]*(i%2==0?1:-1)*.1);
            for(int i=0;i<16;i++)o[n++]=previous[i];
            o[n++]=(float)Math.Sin(time*2*Math.PI/2.4);o[n++]=(float)Math.Cos(time*2*Math.PI/2.4);
            for(int i=0;i<24;i++)o[n++]=(float)Clamp((heights[i]-(q[2]-.2192))*5,-2,2);
            foreach(float value in o)if(float.IsNaN(value)||float.IsInfinity(value))throw new ArithmeticException("Non-finite observation");
            return o;
        }

        public static double[] Targets(float[] action,double vx,double wz,double crouch)
        {
            if(action.Length!=16)throw new ArgumentException("Sai action shape mismatch");
            bool moving=Math.Sqrt(vx*vx+wz*wz)>1e-5;
            var t=new double[16];double down=.172812737-.035*crouch;
            for(int i=0;i<16;i++)
            {
                if(float.IsNaN(action[i])||float.IsInfinity(action[i]))throw new ArithmeticException("Non-finite action");
                action[i]=moving?(float)Clamp(action[i],-1,1):0;
                t[i]=.18*action[i];
            }
            for(int leg=0;leg<4;leg++)
            {
                double side=leg%2==0?1:-1, front=leg<2?1:-1;
                double beta=-front*Math.Acos(Clamp((down*down-.09*.09-.11*.11)/(2*.09*.11),-1,1));
                double theta=-Math.Atan2(.11*Math.Sin(beta),.09+.11*Math.Cos(beta));
                double theta0=front*Math.Atan2(.05,.074833147);
                double beta0=-front*(Math.Atan2(.05,.09797959)+Math.Atan2(.05,.074833147));
                t[4*leg+1]+=side*(theta0-theta);t[4*leg+2]+=side*(beta0-beta);
                t[4*leg+3]=side*((vx-wz*side*.146)/.048+6*action[4*leg+3]);
            }
            return t;
        }
    }
}
