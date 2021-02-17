package example;

import org.jblas.DoubleMatrix;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Stream;

public class Main {
    static SigmoidNetwork sn;
    public static void serialize(Object obj) throws IOException
    {
        FileOutputStream fileOutputStream = new FileOutputStream("net.ser");
        try(ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream))
        {
            objectOutputStream.writeObject(obj);
            objectOutputStream.flush();
        }
        System.out.println("Serialized");
    }

    public static Object deserialize()throws IOException, ClassNotFoundException
    {
        Object obj;


            FileInputStream fileInputStream = new FileInputStream("net.ser");
            try(ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream))
            {
                obj = objectInputStream.readObject();

            }
            return obj;
    }
    public static DoubleMatrix intToDoubleMatrix(int i)
    {
        double[] x = Stream.iterate(0,n -> 0).limit(256).mapToDouble(Double::new).toArray();
        x[i] = 1;
        return new DoubleMatrix(x);
    }
    public static String doubleMatrixToString(DoubleMatrix dm)
    {
        DoubleMatrix tmpOut = sn.feedForward(dm);
        StringBuilder sb = new StringBuilder();
        for (double d : tmpOut.toArray())
        {
            sb.append(d >= 0.5 ? 1 : 0).append(' ');
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
       for (int i = 0; i < 256; i++)
       {
           double[][] io = new double[2][];
           double[] x = new double[256];
           double[] y = new double[8];
           String binary = String.format("%8s",Integer.toBinaryString(i)).replace(' ','0');
           x = Stream.iterate(0,n -> 0).limit(256).mapToDouble(Double::new).toArray();
           y = Arrays.stream(binary.split("")).mapToDouble(Double::parseDouble).toArray();

           x[i] = 1;
           io[0] = x;
           io[1] = y;
           inputOutputs.add(io);
//           sn = (SigmoidNetwork) deserialize();
//           try (Scanner sc = new Scanner(System.in))
//           {
//               while (sc.hasNext())
//               {
//                   int input = Integer.parseInt(sc.next());
//                   System.out.println(doubleMatrixToString(intToDoubleMatrix(input)));
//               }
//           }


       }
        SigmoidNetwork sn = new SigmoidNetwork(256,32,8);
        sn.SGD(inputOutputs,1000,8,15,inputOutputs.subList(0,100));
 //double[] outputs = sn.feedForward(inputs);
        //DoubleMatrix outputs = sn.feedForward(new DoubleMatrix(inputs));

            //System.out.println(outputs.toString("%.0f"));

    }
}
