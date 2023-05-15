/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package redescriptionmining;

import gnu.trove.iterator.TIntIterator;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import static redescriptionmining.SettingsReader.ENCODING;

/**
 *
 * @author matej
 */
public class TargetDistributions {
    /*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
    
    public static void main(String[] args) {
        
        Mappings map=new Mappings();
        map.createIndex("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\Jinput.arff");
        DataSetCreator dat=new DataSetCreator("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\Jinput.arff");
        try{
        dat.readDataset();
        }
        catch(Exception e){
            e.printStackTrace();
        }
        
        HashMap<String,Double> targetLabel=new HashMap<>();
        HashMap<Integer,HashMap<Double,Integer>> distribution=new HashMap<>();
        //map.printMapping();
        
        int test=0;
        if(test==1)
            return;
        
         ApplicationSettings appset=new ApplicationSettings();
        appset.readSettings(new File("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\Settings.set"));
           
        for(int i=0;i<dat.numExamples;i++){
            Double valT=dat.getValue(map.attId.get("CDGLOBAL"), i);
            targetLabel.put(map.idExample.get(i), valT);
            System.out.println("ex: "+map.idExample.get(i)+" value: "+valT);
        }
        
        ReadCLUSRMReds readerReds=new ReadCLUSRMReds();
        readerReds.inputFile=new File("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\redescriptionsGuidedExperimentalIterativeTSMJsuoo5-10ClTD=14NT=60NI=20.rr");
        readerReds.readReds(dat);
        
        for(int i=0;i<readerReds.set.redescriptions.size();i++){
            TIntIterator it=readerReds.set.redescriptions.get(i).elements.iterator();
            HashMap<Double,Integer> tmp=new HashMap<>();
            
            while(it.hasNext()){
                int el=it.next();
                String Selem="\""+el+"\"";
                System.out.println("Element: "+el);
                double val=targetLabel.get(Selem);
                if(!tmp.containsKey(val)){
                    tmp.put(val, 1);
                }
                else{
                    int num=tmp.get(val);
                    num++;
                    tmp.put(val, num);
                }
                    
            }
            distribution.put(i, tmp);
        }
        
        try{
            File output=new File("RedescriptionsDistributions.txt");
         PrintWriter out = new PrintWriter(output.getAbsolutePath());
         
         Iterator<Integer> it=distribution.keySet().iterator();
         
         //while(it.hasNext()){
         for(int i=0;i<readerReds.set.redescriptions.size();i++){
             int rInd=i;//it.next();
             HashMap<Double,Integer> dV=distribution.get(rInd);
             
             Iterator<Double> itD=dV.keySet().iterator();
             int keyNum=0;
             while(itD.hasNext()){
                 double val=itD.next();
                 int count=dV.get(val);
                 //System.out.println("keyNum: "+keyNum+"val: "+val);
                 if(keyNum==0 && val!=0.0){
                     System.out.println("entered missing 0 section");
                     out.write("value: "+0.0+" count: "+0+" ");
                     keyNum++;
                 }
                 
                 if(keyNum==1 && val!=1.0){
                     out.write("value: "+1.0+" count: "+0+" ");
                     keyNum++;
                 }
                 
                 if(keyNum==2 && val!=0.5){
                     out.write("value: "+0.5+" count: "+0+" ");
                     keyNum=0;
                     continue;
                 }
                 
                 
                 out.write("value: "+val+" count: "+count+" ");
                 keyNum=(keyNum+1)%3;
                 
             }
             
             if(keyNum==2)
                 out.write("value: "+0.5+" count: "+0+" ");
             
             out.write("\n");
         }
         
         out.close();
         }
         catch(FileNotFoundException ex){
             ex.printStackTrace();
         }
        
         test=1;
        if(test==1)
            return;
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
        double val=dat.getValue(map.attId.get("BAL"), i);
        double val1=dat.getValue(map.attId.get("M2"), i);
         /*double val2=dat.getValue(map.attId.get("CRED"), i);
          double val3=dat.getValue(map.attId.get("BAL"), i);
           double val4=dat.getValue(map.attId.get("LABOR_PARTICIP_RATE"), i);
            double val5=dat.getValue(map.attId.get("EMPL_POP"), i);*/
        //System.out.println("attr: "+"E1"+" row index: "+i+" value: "+dat.getValue(map.attId.get("E1"), i));
       // POP_GROWTH >= -1.6021 <= 0.851 AND UNEM_F >= 4.3 <= 30.8 AND CRED >= 34.4743 <= 205.3727 AND BAL >= -11.7496 <= 6.9633 AND LABOR_PARTICIP_RATE >= 40.0 <= 64.7 AND EMPL_POP >= 32.5 <= 60.0 
        if(val>=-3.8719 && val<=18.7248 && val!=Double.POSITIVE_INFINITY && val1!=Double.POSITIVE_INFINITY && val1>=131.1325 && val1<=480.3042 /*&& val2!=Double.POSITIVE_INFINITY && val2>=34.4743 && val2<=205.3727 && val3!=Double.POSITIVE_INFINITY && val3>=-11.7496 && val3<=6.9633 && val4!=Double.POSITIVE_INFINITY && val4>=40.0 && val4<=64.7 && val5!=Double.POSITIVE_INFINITY && val5>=32.5 && val5<=60.0*/){
            for(String s: map.exampleId.keySet())
                if(i==map.exampleId.get(s)){
            System.out.println("index i: "+i+" example id: "+s+" value: "+val);
            System.out.println("Example id: "+map.exampleId.get(s));
            System.out.println("Example from id: "+map.idExample.get(i));
            System.out.println(map.idExample.get(i));
                }
         }
        else if(val==Double.POSITIVE_INFINITY && val1==Double.POSITIVE_INFINITY /*&& val2==Double.POSITIVE_INFINITY && val3==Double.POSITIVE_INFINITY && val4==Double.POSITIVE_INFINITY && val5==Double.POSITIVE_INFINITY*/){
            countM++;
            System.out.println("All missing");
            System.out.println("Example from id: "+map.idExample.get(i));
        }
        
        else{
            System.out.println("Value not in range but not infinite...");
             for(String s: map.exampleId.keySet())
                if(i==map.exampleId.get(s)){
            System.out.println("index i: "+i+" example id: "+s+" value: "+val);
            System.out.println("Example id: "+map.exampleId.get(s));
            System.out.println("Example from id: "+map.idExample.get(i));
            System.out.println(map.idExample.get(i));
                }
                }
        }
        System.out.println("Num elem with missing value: "+countM);
        System.out.println("Num countries containing att value: "+(map.exampleId.keySet().size()-countM));
        //int test=1;
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