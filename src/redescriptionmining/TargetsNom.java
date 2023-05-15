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
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import static redescriptionmining.SettingsReader.ENCODING;

/**
 *
 * @author matej
 * creates target class distributions for redescriptions
 */
public class TargetsNom {
    
     public static void main(String[] args) {
        
        Mappings map=new Mappings();
        map.createIndex("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\Jinput.arff");
        HashMap<String,String> targetLabel=new HashMap<>();
        HashMap<Integer,HashMap<String,Integer>> distribution=new HashMap<>();
        DataSetCreator dat=new DataSetCreator("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\Jinput.arff");
        try{
        dat.readDataset();
        }
        catch(Exception e){
            e.printStackTrace();
        }
        
        int countT=0;
        File targ=new File("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\testDataFirstPartClass.txt");
        Path path =Paths.get(targ.getAbsolutePath());
       BufferedReader reader=null;
       try{
       reader=Files.newBufferedReader(path,ENCODING);
       
       
       String line="";
       
       while((line=reader.readLine())!=null){
           countT++;
           String num="\""+countT+"\"";
           targetLabel.put(num, line.trim());
       }
        }
       catch(Exception e){}
       
         ReadCLUSRMReds readerReds=new ReadCLUSRMReds();
       // readerReds.inputFile=new File("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\redescriptionsGuidedExperimentalIterativeTSMJsupp100-300CITD=14NT=60NI=20.rr");
        readerReds.inputFile=new File("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\redescriptionsConstrainedADFirst100-820Targeted");
         readerReds.readReds(dat);
        
        ArrayList<String> posTVal=new ArrayList<>();
        posTVal.add("CN"); posTVal.add("SMC"); posTVal.add("EMCI"); posTVal.add("LMCI"); posTVal.add("AD");
        
        for(int i=0;i<readerReds.set.redescriptions.size();i++){
            TIntIterator it=readerReds.set.redescriptions.get(i).elements.iterator();
            HashMap<String,Integer> tmp=new HashMap<>();
            
            while(it.hasNext()){
                int el=it.next();
                String Selem="\""+el+"\"";
                System.out.println("Element: "+el);
                String val=targetLabel.get(Selem);
                if(!tmp.containsKey(val)){
                    tmp.put(val, 1);
                }
                else{
                    int num=tmp.get(val);
                    num++;
                    tmp.put(val, num);
                }                    
            }
            
            for(String s:posTVal)
                if(!tmp.containsKey(s))
                    tmp.put(s, 0);
            
            distribution.put(i, tmp);
        }
        
        try{
            File output=new File("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\redescriptionsConstrainedADFirst100-820TargetedDist.txt");
         PrintWriter out = new PrintWriter(output.getAbsolutePath());
         
         Iterator<Integer> it=distribution.keySet().iterator();
         
         //Object allTar[]=posTVal.toArray();
         
         //while(it.hasNext()){
         for(int i1=0;i1<readerReds.set.redescriptions.size();i1++){
             int rInd=i1;//it.next();
             HashMap<String,Integer> dV=distribution.get(rInd);
             
             //Iterator<String> itD=dV.keySet().iterator();
             int keyNum=0;
             //while(itD.hasNext()){
               //  String val=itD.next();
             for(int i=0;i<posTVal.size();i++){
                 String val=posTVal.get(i);
                 int count=dV.get(val);
                 //System.out.println("keyNum: "+keyNum+"val: "+val);
                 /*if(keyNum==0 && val!=0.0){
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
                 }*/
                 out.write(/*"value: "+val+" count: "+*/count+" ");
                 //keyNum=(keyNum+1)%3;
                 
             }
             
             out.write("\n");
         }
         
         out.close();
         }
         catch(FileNotFoundException ex){
             ex.printStackTrace();
         }
        
     }
    
}
