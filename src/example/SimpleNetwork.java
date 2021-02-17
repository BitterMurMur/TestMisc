package example;

public class SimpleNetwork {  // сеть на перцептронах
    private final int BIAS = 0;
    private final int WEIGHT = 1;
    private int numLayers;
    private int[] sizes;

    public SimpleNetwork (int...sizes)
    {
        this.sizes = sizes;
        this.numLayers=sizes.length;
    }

    public int[] feedForward(int[] inputs) {
        int[] outputs = null;
        for (int i = 1;i<numLayers;i++)
        {
            int size = sizes[i];              // размер текущего слоя
            int[]z = new int[size];           // результат выполнения z=w*x+b
            outputs = new int[size];
            for (int i2=0;i2<size;i2++)        //прошли по всем нейронам слоя
            {
                for(int i3=0;i3<inputs.length;i3++)
                {
                    z[i2]+=WEIGHT*inputs[i3];
                }
                    z[i2]+=BIAS;
                outputs[i2]=z[i2]>0?1:0;
            }
            inputs = outputs;  // выход одного слоя это вход следующего слоя
        }
        return outputs;
    }
}
