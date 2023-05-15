/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package datasetcreator;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;

/**
 *
 * @author matej
 */
public class CreateArff {
    public static void main(String []args){
       
        /*String path = "C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\DS2019\\Secom";
        String dataInput = path+"\\secom.data";
        String attributeInput = path+"\\secom_labels.data";
        String output = path+"\\secom.arff";*/
        
         /*String path = "C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\DS2019\\SportsArticles";
        String dataInput = path+"\\data.txt";
        String attributeInput = path+"\\data.labels";
        String output = path+"\\sportArt.arff";*/
        
        /*String path = "C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\DS2019\\Theorem proving\\ml-prove";
        String dataInput = path+"\\data.txt";
        String attributeInput = path+"\\data.labels";
        String output = path+"\\theoremProve.arff";*/
        
        String path = "C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\DS2019\\pd_speech_features";
        String dataInput = path+"\\data1.txt";
        String attributeInput = path+"\\attributes.txt";
        String output = path+"\\PDSpeach.arff";
        
        HashMap<Integer,ArrayList<Double>> data = new HashMap<>();
        ArrayList<String> attributes = new ArrayList<>();
        ArrayList<String> entityID = new ArrayList<>();
        
        int attribsAvailable = 1;
     
        File in = new File(dataInput);
        File inAt = new File(attributeInput);
        File out = new File(output);
        
        Path p= Paths.get(in.getAbsolutePath());
        
        
        BufferedReader read = null;
        int count = 0;
        
        try{
            read = Files.newBufferedReader(p);
            
            String line = "";
            int f=0,cnt = 0;
            while((line = read.readLine())!=null){
                if(attribsAvailable == 1 && f==0){
                    //String t[] = line.split("\t");
                     String t[] = line.split(",");
                    for(int i=0;i<t.length;i++)
                        attributes.add(t[i].trim());
                    f=1;
                    continue;
                }
               // String tmp[] = line.split("\t");//secom, sportsArts
                String tmp[] = line.split(",");//theorem proving
                ArrayList<Double> t = new ArrayList<>();
                
                for(int i=0;i<tmp.length;i++){
                    if(tmp[i].contains("+"))
                        tmp[i] = tmp[i].replaceAll("\\+", "");
                    if(i==0){
                       // entityID.add(tmp[i].trim());
                        //continue;
                        entityID.add("t"+(++cnt));
                    }
                    else if(!tmp[i].equals("NaN"))
                           t.add(Double.parseDouble(tmp[i]));
                    else t.add(Double.POSITIVE_INFINITY);
                }
                data.put(++count, t);
            }
            read.close();
        }
        catch(IOException e){
            e.printStackTrace();
        }
        
         /*   p= Paths.get(inAt.getAbsolutePath());
         ArrayList<Integer> targets=new ArrayList<>();
         
          try{
            read = Files.newBufferedReader(p);
            
            String line = "";
            
            while((line = read.readLine())!=null){
                String tmp[] = line.split("\t");
                    targets.add(Integer.parseInt(tmp[0]));
            }
            read.close();
        }
        catch(IOException e){
            e.printStackTrace();
        }*/
        
          try{
          FileWriter fw = new FileWriter(out);
          //fw.write("@RELATION spart\n\n");
          //fw.write("@attribute tid string\n");
           fw.write("@attribute pid string\n");
          
          /*for(int i=0;i<data.get(1).size();i++){
              fw.write("@attribute f"+(i+1)+" string\n");
          }*/
          
         /* for(int i=0;i<data.get(1).size();i++){
              /*if(i==0)
                  fw.write("@attribute tid"+" string\n");*/
              /*else*/ //if(i<data.get(1).size()-1)
          /*       fw.write("@attribute thF"+(i+1)+" numeric\n");
              else fw.write("@attribute class"+" {1,-1}\n\n");
          }*/
          
          for(int i=0;i<attributes.size();i++)
              if(i==0)
                  fw.write("@attribute "+attributes.get(i)+" string\n");
              else fw.write("@attribute "+attributes.get(i)+" numeric\n");
          
          fw.write("@attribute class {0,1}\n\n");
          
          fw.write("@data\n");
          
          String row = "";
          for(int i=1;i<=count;i++){
              row = ""+"\""+entityID.get(i-1)+"\""+",";
              ArrayList<Double> t = data.get(i);
              //row+=i+",";
              for(int j=0;j<t.size();j++)
                        if(t.get(j)!=Double.POSITIVE_INFINITY)
                                row+=t.get(j)+",";
                        else row+="?,";
              row+="\n";
            //  row+=targets.get(i-1)+"\n";
              fw.write(row);
          }
           fw.close();
          }
          catch(IOException e){
              e.printStackTrace();
          }        
    }
}
