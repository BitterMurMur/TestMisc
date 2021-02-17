package example;

import example.Main;
import org.jblas.DoubleMatrix;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;

public class SigmoidNetwork implements Serializable {  // сеть на сигмоиде
    //private final int BIAS = 0;
   // private final int WEIGHT = 1;

    private DoubleMatrix[] weights;
    public DoubleMatrix[] biases;
    private int numLayers;
    private int[] sizes;

    public SigmoidNetwork(int...sizes)
    {
        this.sizes = sizes;
        this.numLayers = sizes.length;
        this.biases = new DoubleMatrix[sizes.length-1];
        this.weights = new DoubleMatrix[sizes.length-1];
        Random random = new Random();
        for(int i = 1;i < sizes.length;i++) // заполение смещений рандомными значениями
        {
            double[][]tmp = new double[sizes[i]][];
            for (int i2 = 0;i2 < sizes[i];i2++)
            {
                //double[]b = new double[] {1}; // константа для всех смещений пока что
                double[]b = new double[] {random.nextGaussian() };
                tmp[i2] = b;
            }
            biases[i-1] = new DoubleMatrix(tmp);
        }
        for(int i = 1;i < sizes.length;i++) // заполение весов рандомными значениями
        {
            double[][]tmp = new double[sizes[i]][];
            for (int i2 = 0;i2 < sizes[i];i2++)
            {
                double[]w = new double[sizes[i-1]] ;
                for(int i3 = 0;i3 < sizes[i-1];i3++)
                {
                   // w[i3] = 1;        // потом заменть на рандомные значения
                    w[i3] = random.nextGaussian();
                }
                tmp[i2] = w;
            }
            weights[i-1] = new DoubleMatrix(tmp);
        }
    }

//    public double[] feedForward(double[] inputs) {
//        double[] outputs = null;
//        for (int i = 1;i<numLayers;i++)
//        {
//            int size = sizes[i];              // размер текущего слоя
//            int[]z = new int[size];           // результат выполнения z=w*x+b
//            outputs = new double[size];
//            for (int i2=0;i2<size;i2++)        //прошли по всем нейронам слоя
//            {
//                for(int i3=0;i3<inputs.length;i3++)
//                {
//                    z[i2]+=WEIGHT*inputs[i3];
//                }
//                    z[i2]+=BIAS;
////                outputs[i2]=z[i2]>0?1:0;
//                outputs[i2] = 1/(1+Math.exp(z[i2]));
//            }
//            inputs = outputs;  // выход одного слоя это вход следующего слоя
//        }
//        return outputs;
//    }
    public DoubleMatrix feedForward(DoubleMatrix a)
    {
        for (int i = 0;i < numLayers-1;i++)
        {
            double[]z = new double[weights[i].rows];
            for(int i2 = 0;i2 < weights[i].rows;i2++)
            {
                z[i2] = weights[i].getRow(i2).dot(a)+biases[i].get(i2);
            }
            DoubleMatrix output = new DoubleMatrix(z);
            a = sigmoid(output);
        }
        return a;
    }

    private DoubleMatrix sigmoid(DoubleMatrix z) { // для каждого элемента вектора z вычисляем сигмоиду
    double output[] = new double[z.length];
    for(int i = 0;i < output.length;i++)
    {
        output[i] = 1/(1+Math.exp(-z.get(i))); // сигмоида
    }
    return new DoubleMatrix(output);
    }

    private int evaluete (List<double[][]> testData)
    {
        int sum = 0;
        for (double[][] inputOutput : testData)
        {
            DoubleMatrix x = new DoubleMatrix(inputOutput[0]);
            DoubleMatrix y = new DoubleMatrix(inputOutput[1]);
            DoubleMatrix netOutput = feedForward(x);
            StringBuilder sb = new StringBuilder();
            StringBuilder sb1 = new StringBuilder();
            for (double d : netOutput.toArray())
            {
                sb.append(d >= 0.5 ? 1 : 0);
            }
            for (double d : y.toArray())
            {
                sb1.append(d >= 0.5 ? 1 : 0);
            }
            if (Integer.parseInt(sb.toString(),2) == Integer.parseInt(sb1.toString(),2))
            {
                sum++;
            }
        }
        return sum;
    }

     public DoubleMatrix[][] backProp(double[][] inputoutputs) {
        DoubleMatrix[] nableB = new DoubleMatrix[biases.length]; // вектор частных производных (градиент)
        DoubleMatrix[] nableW = new DoubleMatrix[weights.length]; // вектор частных производных (градиент)
        for (int i = 0; i < nableB.length;i++)
        {
            nableB[i] = new DoubleMatrix(biases[i].getRows(),biases[i].getColumns());
        }
        for (int i = 0; i < nableW.length;i++)
        {
            nableW[i] = new DoubleMatrix(weights[i].getRows(),weights[i].getColumns());
        }
        // метод feedForward
        DoubleMatrix activation = new DoubleMatrix(inputoutputs[0]);
        DoubleMatrix[] activations = new DoubleMatrix[numLayers];
        activations[0] = activation;
        DoubleMatrix[] zs = new DoubleMatrix[numLayers - 1];
        for (int i = 0;i < numLayers-1;i++)
        {
            double [] scalars = new double[weights[i].rows];
            for (int i2 = 0;i2 < weights[i].rows;i2++)
            {
                scalars[i2] = weights[i].getRow(i2).dot(activation)+biases[i].get(i2);
            }
            DoubleMatrix z = new DoubleMatrix(scalars);
            zs[i] = z;
            activation = sigmoid(z);
            activations[i+1] = activation;
        }
        // back pass метод обратного прохождения по сети
        DoubleMatrix output = new DoubleMatrix(inputoutputs[1]);
        DoubleMatrix delta = costDerivative(activations[activations.length - 1],output).mul(sigmoidPrime(zs[zs.length - 1])); // функция ошибок для последнего слоя
        nableB[nableB.length - 1] = delta; // для расчета смещений
        nableW[nableW.length - 1] = delta.mmul(activations[activations.length - 2].transpose()); // для расчета весов
        for (int l = 2; l < numLayers; l++)
        {
            DoubleMatrix z = zs[zs.length-l];
            DoubleMatrix sp = sigmoidPrime(z);
            delta = weights[weights.length + 1 - l].transpose().mmul(delta).mul(sp); // для расчета функции ошибок низлежащих слоев
            nableB[nableB.length - l] = delta;
            nableW[nableW.length - l] = delta.mmul(activations[activations.length - 1 - l].transpose());
        }
        return new DoubleMatrix[][] {nableB,nableW};
    }

    private DoubleMatrix sigmoidPrime(DoubleMatrix z) {
        return sigmoid(z).mul(sigmoid(z).rsub(1));
    }

    private DoubleMatrix costDerivative(DoubleMatrix outputActivations, DoubleMatrix output) {
        return outputActivations.sub(output);
    }

    public void SGD(List<double[][]> trainingData, int epochs, int miniBatchSize, double eta, List<double[][]> testData) // стохастический спуск
    {
        int nTest = 0;

        int n = trainingData.size(); // количество посылаемого для обучения материала

        if (testData != null)
        {
            nTest = testData.size();
        }

        for (int i2 = 0; i2 < epochs; i2++)
        {
            Collections.shuffle(trainingData); // рандомизация выборки вариантов
            List<List<double[][]>> miniBatches = new ArrayList<>(); // лист подгрупп
            for (int i3 = 0; i3 < n; i3 +=miniBatchSize)
            {
                miniBatches.add(trainingData.subList(i3,i3+miniBatchSize));
            }
            for (List<double[][]> miniBatch:miniBatches)
            {
                updateMiniBatch(miniBatch, eta); //обновляем каждую мини группу

            }
            if (testData != null)
            {
                int e = evaluete(testData);
                System.out.println(String.format("Epoch %d: %d/ %d" ,i2,e,nTest));
                if (e >= nTest*0.95)
                {
                    try
                    {
                        Main.serialize(this);
                    }
                    catch (IOException e1)
                    {
                        e1.printStackTrace();
                    }
                    break;
                }
            }else
            System.out.println(String.format("Epoch %d complete",i2));
        }
    }

    private void updateMiniBatch(List<double[][]> miniBatch, double eta) {
        DoubleMatrix[] nablaB = new DoubleMatrix[biases.length];
        DoubleMatrix[] nablaW = new DoubleMatrix[weights.length];

        for(int i = 0; i < nablaB.length; i++)
        {
            nablaB[i]=new DoubleMatrix(biases[i].getRows(),biases[i].getColumns());
        }
        for(int i = 0; i < nablaW.length; i++)
        {
            nablaW[i]=new DoubleMatrix(weights[i].getRows(),weights[i].getColumns());
        }
        for (double[][] inputOutput : miniBatch)
        {
            DoubleMatrix[][] deltas = backProp(inputOutput);

            DoubleMatrix[] deltaNablaB = deltas[0];
            DoubleMatrix[] deltaNablaW = deltas[1];
            for (int i = 0; i < nablaB.length;i++)
        {
            nablaB[i] = nablaB[i].add(deltaNablaB[i]);

        }
            for (int i = 0; i < nablaW.length;i++)
            {
                nablaW[i] = nablaW[i].add(deltaNablaW[i]);

            }
        }
        for (int i = 0; i < biases.length; i++)
        {
            biases[i] = biases[i].sub(nablaB[i].mul(eta/miniBatch.size()));
        }
        for (int i = 0; i < weights.length; i++)
        {
           weights[i] = weights[i].sub(nablaW[i].mul(eta/miniBatch.size()));
        }
    }
}
