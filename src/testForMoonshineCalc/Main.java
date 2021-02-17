package testForMoonshineCalc;

import org.jblas.DoubleMatrix;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Stream;

import static testForMoonshineCalc.ArrayInit.ostMass;
import static testForMoonshineCalc.ArrayInit.spiritPercentsMass;

class Main {
    private static SigmoidNetwork sn;
     static void serialize(Object obj) throws IOException
    {
        FileOutputStream fileOutputStream = new FileOutputStream("net_1.ser");
        try(ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream))
        {
            objectOutputStream.writeObject(obj);
            objectOutputStream.flush();
        }
        System.out.println("Serialized");
    }

    static Object deserialize()throws IOException, ClassNotFoundException
    {
        Object obj;


            FileInputStream fileInputStream = new FileInputStream("net_1.ser");
            try(ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream))
            {
                obj = objectInputStream.readObject();

            }
            return obj;
    }
    static DoubleMatrix intToDoubleMatrix(int i)
    {
        double[] x = Stream.iterate(0,n -> 0).limit(ostMass.length).mapToDouble(Double::new).toArray();
        x[i] = ostMass[i];
        return new DoubleMatrix(x);
    }
    static String doubleMatrixToString (DoubleMatrix dm)
    {
        DoubleMatrix tmpOut = sn.feedForward(dm);
        StringBuilder sb = new StringBuilder();
        for (double d : tmpOut.toArray())
        {
            sb.append(' ');
        }
        return sb.toString();
    }
    public static void main(String...args) throws ClassNotFoundException, IOException
    {
        //example.SigmoidNetwork sn = new example.SigmoidNetwork(2,3,2);
//        example.SigmoidNetwork sn = new example.SigmoidNetwork(1,1);
//       double [] inputs={0};
//       double [] outputs = {0};
//       double [][] inputoutputs = new double[][]{inputs,outputs};
//       DoubleMatrix[][] deltas=sn.backProp(inputoutputs);
//       for (int i = 0; i < sn.biases.length; i++)
//       {
//           sn.biases[i] = sn.biases[i].sub(deltas[0][i].mul(4));
//       }
//        System.out.println("Done");
       List<double[][]> inputOutputs = new ArrayList<>();
       for (int i = 0; i < ostMass.length; i++)
       {
           double[][] io = new double[2][];
           double[] x = new double[ostMass.length];
           double[] x1 = new double[spiritPercentsMass.length];
           double[] y = new double[spiritPercentsMass.length];
           x = ostMass;
           y = spiritPercentsMass;
           //y = Stream.iterate(0,n -> 0).limit(spiritPercentsMass.length).mapToDouble(Double::new).toArray();

//           x[i] = ostMass[i];
//           y[i] = spiritPercentsMass[i];
           io[0] = x;
           io[1] = y;
           //io[2] = y;

           inputOutputs.add(io);


//           sn = (SigmoidNetwork) deserialize();
//            try (Scanner sc = new Scanner(System.in))
//            {
//                while (sc.hasNext())
//                {
//                    int input = Integer.parseInt(sc.next());
//                    System.out.println(doubleMatrixToString(intToDoubleMatrix(input)));
//                }
//            }
       }
        sn = new SigmoidNetwork(46,50,1); //пока не понятно как научить хавать  массив даблов
        sn.SGD(inputOutputs,1000,2,1,inputOutputs);
       //double[] outputs = sn.feedForward(inputs);
        //DoubleMatrix outputs = sn.feedForward(new DoubleMatrix(inputs));

            //System.out.println(outputs.toString("%.0f"));

    }
}
