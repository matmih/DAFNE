/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package redescriptionmining;

import java.util.Random;
import org.rosuda.JRI.Rengine;

/**
 *
 * @author Matej
 */
public class FeatureConstructionWithBerouta {
    public static void main(String args[]){
        //add 100 runs
        //load input data
        //create train/test set
        //create rules - these should be supervised, see how to do this???
        //Ripper, MD5 supervised rules
        //train RF of PCT directly on the input data, using target labels
        //transform forest to rules and use these rules as features
        //create new createSettings function????
        //create subgroups -> use general implementation of CN2SD
        //create redescriptions -> already implemented
        //create data containing newly created features, train/validation + test
        //add actual R code
        //run the R function and create the data
        //make sure algo evaluation can be done in parallel 
        //run all algos on the created datasets and save the result
        //framework can deal with Categorical labels -> classification task
        //measures: AUC/AUPRC accross 100 runs for orig, orig+Suprules, orig+Descrules, orig+subg, orig+reds, allFeatures
        //percentages: perc of all Suprules in boruta, perc of all reds in boruta, perc of all subg in boruta, perc of all Descrules
        Rengine engine = Rengine.getMainEngine();
        if(engine == null){
            engine=new Rengine (new String [] {"--vanilla"}, false, null);
        if (!engine.waitForR())
        {
            System.out.println ("Cannot load R");
            return;
        }
    }
        
     engine.eval("tmp<-'Proba'");
     String t = engine.eval("tmp").toString();
     System.out.println("t: "+t);
     engine.eval("tmp<-paste(tmp,'Proba',sep = \"\")");
        
     String rFunkcija = "C:/Users/Matej/OneDrive/Dokumenti/compFeatures2.R";
     
     // String input = "F:/Matej Dokumenti/Redescription mining with CLUS/FeatureConstructionTrainValidation.arff";
      //String inputTest = "F:/Matej Dokumenti/Redescription mining with CLUS/FeatureConstructionTest.arff";
      String input = "F:/Matej Dokumenti/Redescription mining with CLUS/TrainValidationAllFeatures.arff";
      String inputTest = "F:/Matej Dokumenti/Redescription mining with CLUS/TestAllFeatures.arff";
      String dataset = "Abalone";
      String output = "F:/Matej Dokumenti/Redescription mining with CLUS/";
     
      engine.eval("input<-'"+input+"'");
      String s = engine.eval("input").asString();
       System.out.println(s);
      engine.eval("inputTest<-'"+inputTest+"'");
      engine.eval("dataset<-'"+dataset+"'");
      engine.eval("output<-'"+output+"'");
      
      //engine.eval("library('farff',lib.loc = \"C:/Users/Matej/OneDrive/Dokumenti/R/win-library/4.1\")");
      engine.eval("library('Boruta',lib.loc = \"C:/Users/Matej/OneDrive/Dokumenti/R/win-library/4.1\")");
      engine.eval("library('foreign')");
      engine.eval("library('ranger',lib.loc = \"C:/Users/Matej/OneDrive/Dokumenti/R/win-library/4.1\")");
      engine.eval("library('randomForest',lib.loc = \"C:/Users/Matej/OneDrive/Dokumenti/R/win-library/4.1\")");
      engine.eval("library('varSelRF',lib.loc = \"C:/Users/Matej/OneDrive/Dokumenti/R/win-library/4.1\")");
      //engine.eval("library('digest',lib.loc = \"C:/Users/Matej/OneDrive/Dokumenti/R/win-library/4.1\")");
      //engine.eval("library('entropy',lib.loc = \"C:/Users/Matej/OneDrive/Dokumenti/R/win-library/4.1\")");
      //engine.eval("library('mlbench',lib.loc = \"C:/Users/Matej/OneDrive/Dokumenti/R/win-library/4.1\")");
      //engine.eval("library('rpart',lib.loc = \"C:/Users/Matej/OneDrive/Dokumenti/R/win-library/4.1\")");
     // engine.eval("library('rWeka',lib.loc = \"C:/Users/Matej/OneDrive/Dokumenti/R/win-library/4.1\")");
      //engine.eval("library('FSelector',lib.loc = \"C:/Users/Matej/OneDrive/Dokumenti/R/win-library/4.1\")");
      String ver = engine.eval("version").toString();
      System.out.println("version: "+ver);
        engine.eval("source('" +rFunkcija + "')");
        engine.eval("X <- lsf.str()");
        String t2 = engine.eval("as.vector(X)").toString();
      t = engine.eval("tmp").toString();
      System.out.println("t1: "+t);
      System.out.println("t2: "+t2);
      String ss = engine.eval("computeFeaturesAndDatasetsRJS").asString();
      System.out.println(ss);
      //engine.eval("install.packages(\"farff\")");
     
    //  String pr = engine.eval("print(readARFF)").toString();
     // System.out.println("pr: "+pr);
    // engine.eval("data<-read.arff(input)");
   //  engine.eval("data1<-data[,-1];");
      //String r1 = engine.eval("Boruta").asString();
     //String r1 = engine.eval("Boruta.Test<-Boruta(Rings ~.,data = data1,doTrace = 2,ntree = 500)").asString();
    // System.out.println("r1: "+r1);
      String pr1 = engine.eval("search()").toString();
      System.out.println("pr1: "+pr1);
      /*String r =*/ engine.eval("computeFeaturesAndDatasetsRJS(input,inputTest,dataset,output)");//.asString(); 
     // System.out.println("r: "+r);
       String rFunkcija1 = "F:/Matej Dokumenti/Ostalo/crtajHistogram.R";
        engine.eval("source('" +rFunkcija1 + "')");
        double brojevi[] = new double[1000];
      Random rand = new Random();
      for(int i=0;i<1000;i++)
        brojevi[i]=rand.nextGaussian();
      
      String newRVektor = "c(";
          for(int i=0;i<brojevi.length;i++){
              if(i+1<brojevi.length)
                  newRVektor+=brojevi[i]+",";
              else  newRVektor+=brojevi[i]+")";
          }
       engine.eval("x<-"+newRVektor);
       String c = "F:/Matej Dokumenti/Redescription mining with CLUS/normalni.png";
       engine.eval("inputCrtaj<-'"+c+"'");
        //String ss1 = engine.eval("crtajHistogram(x,\"F:/Matej Dokumenti/Redescription mining with CLUS/normalni.png\")").asString();
       String ss1 = engine.eval("crtajHistogram(x,inputCrtaj)").asString();
        System.out.println("ss1: "+ss1);
          engine.eval("dev.off()");
      engine.end();
    }
}
