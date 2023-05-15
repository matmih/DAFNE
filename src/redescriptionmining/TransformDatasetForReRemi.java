/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package redescriptionmining;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import static redescriptionmining.SettingsReader.ENCODING;
import org.apache.commons.math3.distribution.BinomialDistribution;
/**
 *
 * @author matej
 */
public class TransformDatasetForReRemi {
    
    public static void main(String[] args) {

        double prob=4.0/(199*199);
        BinomialDistribution dist=new BinomialDistribution(199,prob);
        
        System.out.println("Probability: "+dist.cumulativeProbability(2));
        System.out.println("p-value: "+(1-dist.cumulativeProbability(2)));
        
        prob=(190.0/199000.0)*(199.0/199000.0);
        //prob=1.0;
        dist=new BinomialDistribution(199000,prob);
        
        System.out.println("Probability: "+dist.cumulativeProbability(19));
        System.out.println("p-value: "+(1.0-dist.cumulativeProbability(19)));
        
        double d=Math.pow(10, -17);
        System.out.println("Small number "+d);
        
        int test=1;
        
        /*if(test==1)
            return;*/
        
        Mappings map=new Mappings();
        map.createIndex("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\Jinput.arff");
        DataSetCreator dat=new DataSetCreator("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\Jinput.arff");
        try{
        dat.readDataset();
        }
        catch(Exception e){
            e.printStackTrace();
        }
        
        map.printMapping();
        
         ApplicationSettings appset=new ApplicationSettings();
        appset.readSettings(new File("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\Settings.set"));
        
        /*String input1=appset.outFolderPath+"\\view2.out";
           RuleReader rr1=new RuleReader();
            RuleReader rr2=new RuleReader();
           rr1.extractRules(input1,map,dat,appset);
           String input2=appset.outFolderPath+"\\view2tmp.out";
           rr2.extractRules(input2, map, dat, appset);
           rr1.addnewRulesC(rr2, appset.numnewRAttr, 0);
           
           int countM1=0;
          Rule r=null;
          
          for(int i=0;i<rr1.rules.size();i++)
              if(rr1.rules.get(i).rule.contentEquals("DATE = true")){
                  r=rr1.rules.get(i);
                  break;
              }*/
           
          /* for(int i=0;i<dat.numExamples;i++){
                String valT=dat.getValueCategorical(map.attId.get("DATE"), i);
                if(valT.contentEquals("true")) {
            //for(String s: map.exampleId.keySet())
            //    if(i==map.exampleId.get(s)){
           /* System.out.println("index i: "+i+" example id: "+s+" value: "+val);
            System.out.println("Example id: "+map.exampleId.get(s));
            System.out.println("Example from id: "+map.idExample.get(i));*/
         /*   System.out.println("index: "+i);
            System.out.println("Example: "+map.idExample.get(i));
            System.out.println("Value: "+valT);
            if(!r.elements.contains(i))
                System.out.println("Element missing");
          //      }
         }
        else// if(val==Double.POSITIVE_INFINITY)
            countM1++;
        }*/
           
    
          /* System.out.println("Num elem with missing value: "+countM1);
        System.out.println("Num countries containing att value: "+(map.exampleId.keySet().size()-countM1));
           
           if(test==1)
               return;*/
        
     //  t7~ >= 19.8 <= 21.125 AND t7+ >= 22.8 <= 24.3  
        System.out.println("Attr size: "+map.attId.size());
        System.out.println("Elements size: "+map.exampleId.size());
        int countM=0;
        
      /*  for(int i=0;i<dat.numExamples;i++){
        //double val=dat.getValue(map.attId.get("t7+"), i);
            String val=dat.getValueCategorical(map.attId.get("IFIPCongress(1)"), i);
            String val1=dat.getValueCategorical(map.attId.get("HPDC"), i);
            String val2=dat.getValueCategorical(map.attId.get("SOFSEM"), i);
            /*String val3=dat.getValueCategorical(map.attId.get("ICALP"), i);
            String val4=dat.getValueCategorical(map.attId.get("COCOON"), i);
            String val5=dat.getValueCategorical(map.attId.get("IFIPTCS"), i);*/
       /*     String val11=dat.getValueCategorical(map.attId.get("HugoDeMan"), i);
            String val12=dat.getValueCategorical(map.attId.get("JackDongarra"), i);
            String val13=dat.getValueCategorical(map.attId.get("IanT.Foster"), i);
            String val14=dat.getValueCategorical(map.attId.get("SureshSingh"), i);
            String val15=dat.getValueCategorical(map.attId.get("ProsenjitBose"), i);
            //String val16=dat.getValueCategorical(map.attId.get("ErikD.Demaine"), i);
        //System.out.println("attr: "+"E1"+" row index: "+i+" value: "+dat.getValue(map.attId.get("E1"), i));
        if((val.contentEquals("false") && val1.contentEquals("false")&&val2.contentEquals("false")/*&& val3.contentEquals("false")&& val4.contentEquals("false")&& val5.contentEquals("false")) && (val11.contentEquals("false")&& val12.contentEquals("false")&& val13.contentEquals("false")&& val14.contentEquals("false")&& val15.contentEquals("false")/*&& val16.contentEquals("false"))){*/
            //for(String s: map.exampleId.keySet())
            //    if(i==map.exampleId.get(s)){
           /* System.out.println("index i: "+i+" example id: "+s+" value: "+val);
            System.out.println("Example id: "+map.exampleId.get(s));
            System.out.println("Example from id: "+map.idExample.get(i));*/
       /*     System.out.println("index: "+i);
            System.out.println("Example: "+map.idExample.get(i));
            System.out.println("Value: "+val);
          //      }
         }
        else// if(val==Double.POSITIVE_INFINITY)
            countM++;
        }*/
      //  E/I39 >= 0.0 <= 3.221 AND E85 >= 0.0 <= 2.0 
        for(int i=0;i<dat.numExamples;i++){
            String val=dat.getValueCategorical(map.attId.get("ITCC"), i);
            String val1=dat.getValueCategorical(map.attId.get("SEBD"), i);
            String val2=dat.getValueCategorical(map.attId.get("WETICE"), i);
            String val3=dat.getValueCategorical(map.attId.get("GiuseppeManco"), i);
            String val4=dat.getValueCategorical(map.attId.get("AlbertoMartelli"), i);
            String val5=dat.getValueCategorical(map.attId.get("ElioMasciari"), i);
            //String val6=dat.getValueCategorical(map.attId.get("ElioMasciari"), i);
            //String val5=dat.getValueCategorical(map.attId.get("SiegfriedHandschuh"), i);
        //double val=dat.getValue(map.attId.get("MORT"), i);
        //double val1=dat.getValue(map.attId.get("POP_64"), i);
        /* double val2=dat.getValue(map.attId.get("POP_GROWTH"), i);
          double val3=dat.getValue(map.attId.get("E/I66"), i);
           double val4=dat.getValue(map.attId.get("LABOR_PARTICIP_RATE"), i);
            double val5=dat.getValue(map.attId.get("EMPL_POP"), i);*/
        //System.out.println("attr: "+"E1"+" row index: "+i+" value: "+dat.getValue(map.attId.get("E1"), i));
       // POP_GROWTH >= -1.6021 <= 0.851 AND UNEM_F >= 4.3 <= 30.8 AND CRED >= 34.4743 <= 205.3727 AND BAL >= -11.7496 <= 6.9633 AND LABOR_PARTICIP_RATE >= 40.0 <= 64.7 AND EMPL_POP >= 32.5 <= 60.0 
        //if(/*val>=69.0 &&*/ val<=95.5 && val!=Double.POSITIVE_INFINITY /*&& val1!=Double.POSITIVE_INFINITY && val1>=15.3044 && val1<=17.8169 /*&& val2!=Double.POSITIVE_INFINITY && val2>=1.8913 /*&& val2<=0.963 /*&& val3!=Double.POSITIVE_INFINITY && val3>=0.308 && val3<=6.872 /*&& val4!=Double.POSITIVE_INFINITY && val4>=40.0 && val4<=64.7 && val5!=Double.POSITIVE_INFINITY && val5>=32.5 && val5<=60.0*/){
          if((val.toLowerCase().equals("true") && val1.toLowerCase().equals("true") && val2.equals("false")) && (((val3.toLowerCase().equals("true") && val4.toLowerCase().equals("false"))) || (val3.toLowerCase().equals("false") && val5.toLowerCase().equals("true")) ) /*&& val4.equals("true")*/){  
            for(String s: map.exampleId.keySet())
                if(i==map.exampleId.get(s)){
            System.out.println("indexCont i: "+i+" example id: "+s+" value: "+val);
            System.out.println("Example id: "+map.exampleId.get(s));
            System.out.println("Example from id: "+map.idExample.get(i));
            System.out.println(map.idExample.get(i));
                }
         }
        //else if(val==Double.POSITIVE_INFINITY /*&& val1==Double.POSITIVE_INFINITY /*&& val2==Double.POSITIVE_INFINITY && val3==Double.POSITIVE_INFINITY && val4==Double.POSITIVE_INFINITY && val5==Double.POSITIVE_INFINITY*/){
       //     countM++;
       //     System.out.println("All missing");
        //    System.out.println("Example from id: "+map.idExample.get(i));
       // }
        
     //   else{
      //      System.out.println("Value not in range but not infinite...");
      //       for(String s: map.exampleId.keySet())
      //          if(i==map.exampleId.get(s)){
      //      System.out.println("index i: "+i+" example id: "+s+" value: "+val);
      //      System.out.println("Example id: "+map.exampleId.get(s));
        //    System.out.println("Example from id: "+map.idExample.get(i));
        //    System.out.println(map.idExample.get(i));
         //       }
         //       }
        }
        //System.out.println("Num elem with missing value: "+countM);
        //System.out.println("Num countries containing att value: "+(map.exampleId.keySet().size()-countM));
        if(test==1)
            return;
        
        
        ArrayList<ArrayList<String>> data=new ArrayList<>();
        String inputS,output1,output2;
        int count=0;
        
        inputS="C:\\Users\\matej\\Documents\\NetBeansProjects\\Redescription mining Opt\\UNCTADall3transf.txt";
        BufferedReader reader;
        
           try {
      File input=new File(inputS);
      Path path =Paths.get(input.getAbsolutePath());
      System.out.println("Path: "+input.getAbsolutePath());
      reader = Files.newBufferedReader(path,ENCODING);
      String line = null;
      while ((line = reader.readLine()) != null) {
          ArrayList<String> tmp=new ArrayList<>();
          String elem[]=line.split(" ");
          System.out.println("elem size: "+elem.length);
          for(int i=0;i<elem.length;i++)
                tmp.add(elem[i]);
          
          data.add(tmp);
      }
      reader.close();
         }catch(IOException ioe)
            {
              System.err.println("IOException: " + ioe.getMessage());
            }
           
           System.out.println("Reading complete!");
           System.out.println("Read rows: "+data.size());
        
        try{
         FileWriter fw = new FileWriter("C:\\Users\\matej\\Documents\\NetBeansProjects\\Redescription mining Opt\\"+"unctad.densenum");
         FileWriter fw1 = new FileWriter("C:\\Users\\matej\\Documents\\NetBeansProjects\\Redescription mining Opt\\"+"wb.densenum");
         
         for(int j=0;j<data.get(0).size();j++)
             for(int i=0;i<data.size();i++){
                 if(j<312){
                     if((i+1)<data.size())
                    fw.write(data.get(i).get(j)+" ");
                     else
                         fw.write(data.get(i).get(j)+"\n"); 
                 }
                 else{
                    if((i+1)<data.size())
                    fw1.write(data.get(i).get(j)+" ");
                     else
                         fw1.write(data.get(i).get(j)+"\n"); 
                 }
             }
         
         fw.close();
         fw1.close();
        }
        catch(Exception e)
                {e.printStackTrace();}
        
    }
    
}
