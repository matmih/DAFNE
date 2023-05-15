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
import java.util.HashMap;
import java.util.HashSet;
import static redescriptionmining.ApplicationSettings.ENCODING;

/**
 *
 * @author matej
 */
public class ParsersSSB {
    
     public static void main(String[] args) {
     
         
         BufferedReader reader;
         String pathString="C:\\Users\\matej\\Downloads\\SSB data\\ssbOrg.csv";
         HashMap<String,String> ssb=new HashMap<>();
         HashSet<String> organisms=new HashSet<>();
         HashMap<String,ArrayList<Integer>> ssCOG=new HashMap<>();
         HashMap<String,ArrayList<Integer>> ssCOGShort=new HashMap<>();
         HashMap<String,Integer> indexMap=new HashMap<>();
         HashMap<Integer,String> inverseIndexMap=new HashMap<>();
         
         try {
      File input=new File(pathString);
      Path path =Paths.get(input.getAbsolutePath());
      System.out.println("Path: "+input.getAbsolutePath());
      reader = Files.newBufferedReader(path,ENCODING);
      String line = null;
      int fl=0;
      int NE=0;
      
      while ((line = reader.readLine()) != null) {
          
          String elem[]=line.split(";");
          //System.out.println("Elem size: "+elem.length);
          //System.out.println("Num elements: "+(++NE));
         // System.out.println(elem[0].trim());
          
          if(ssb.containsKey(elem[0].trim())){
              System.out.println(elem[0].trim());
              System.out.println(elem[0]);
              System.out.println();
          }
          
          ssb.put(elem[0].trim(), elem[1].trim());
          organisms.add(elem[1].trim());

      }
      reader.close();
         }catch(IOException ioe)
            {
              System.err.println("IOException: " + ioe.getMessage());
            } 
         
         System.out.println("Number of ssb: "+ssb.keySet().size());
         
         try{
         FileWriter fw = new FileWriter("C:\\Users\\matej\\Downloads\\SSB data\\bacteria.arff");
         fw.write("@RELATION SSB");
         fw.write("\n\n");
         fw.write("@ATTRIBUTE sid       "+"string\n");
         for(String s:organisms)
             fw.write("@ATTRIBUTE "+s+"       "+"{1,0}\n");
         
                      
             fw.write("\n");
             fw.write("@DATA\n\n");
             
         for(String SSB:ssb.keySet()){
              int orgSize=organisms.size(), count=0;
             fw.write("\""+SSB+"\",");
             for(String s:organisms){
                 if(ssb.get(SSB).equals(s)){
                     if(count+1<orgSize)
                     fw.write("1,");
                     else fw.write("1\n");
                     count++;
                 }
                 else{
                     if(count+1<orgSize)
                         fw.write("0,");
                     else
                         fw.write("0\n");
                     count++;
                 }
             }
             
            /* fw.write("\""+(i+1)+"\",");
             
             for(int j=0;j<tmp.size();j++){
                 if(j+1<tmp.size()){
                     if(tmp.get(j)!=Double.POSITIVE_INFINITY)
                 fw.write(tmp.get(j)+",");
                     else fw.write("?"+",");
                 }
                 else{
                     if(tmp.get(j)!=Double.POSITIVE_INFINITY)
                     fw.write(tmp.get(j)+"\n");
                     else fw.write("?"+"\n");
                 }
             }*/
         }
         fw.close();
        }
        catch(Exception e){
            e.printStackTrace();
        }
         
         
     }
    
}
