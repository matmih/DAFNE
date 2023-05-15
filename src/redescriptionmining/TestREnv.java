/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package redescriptionmining;

import org.rosuda.JRI.Rengine;

/**
 *
 * @author Matej
 */
public class TestREnv {
    public static void main(String args[]) throws InterruptedException{
         Rengine engine = new Rengine (new String [] {"--vanilla"}, false, null);
        // Rengine engine1 = new Rengine (new String [] {"--vanilla"}, false, null);
     Thread.sleep(10000);
          engine.eval("M1<-matrix(data = 1, nrow = 100, ncol = 100)");
          String t = engine.eval("M1").toString();
          System.out.println("t gen: "+t);
          engine.eval("M1<-M1+M1");
           t = engine.eval("M1").toString();
          System.out.println("t res: "+t);
        // engine1.eval("tmp<-'Proba1'");
          t = engine.eval("M1").toString();
          System.out.println("t res: "+t);
          engine.end();
         // engine1.end();
    }
}
