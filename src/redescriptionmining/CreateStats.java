/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package redescriptionmining;

import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

/**
 *
 * @author Matej
 */
public class CreateStats {
    public static void main(String args[]){
        ArrayList<ArrayList<Double>> mlp = new ArrayList<>();
        ArrayList<ArrayList<Double>> lmt = new ArrayList<>();
        ArrayList<ArrayList<Double>> nb = new ArrayList<>();
        ArrayList<ArrayList<Double>> dst = new ArrayList<>();
        ArrayList<ArrayList<Double>> log = new ArrayList<>();
        ArrayList<ArrayList<Double>> ks = new ArrayList<>();
        ArrayList<ArrayList<Double>> rf600 = new ArrayList<>();
        ArrayList<ArrayList<Double>> j48 = new ArrayList<>();
        
        for(int i=0;i<14;i++){
            mlp.add(new ArrayList<>());
            lmt.add(new ArrayList<>());
            nb.add(new ArrayList<>());
            dst.add(new ArrayList<>());
            log.add(new ArrayList<>());
            ks.add(new ArrayList<>());
            rf600.add(new ArrayList<>());
            j48.add(new ArrayList<>());       
        }
        
        ArrayList<Double> rulePerc = new ArrayList<>();
        ArrayList<Double> suprulePerc = new ArrayList<>();
        ArrayList<Double> subgPerc = new ArrayList<>();
        ArrayList<Double> redsPerc = new ArrayList<>();
        ArrayList<Double> rulePercTest = new ArrayList<>();
        ArrayList<Double> suprulePercTest = new ArrayList<>();
        ArrayList<Double> subgPercTest = new ArrayList<>();
        ArrayList<Double> redsPercTest = new ArrayList<>();
        
        
        String inputPath = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\Eksperimenti konstrukcija značajki\\NFMCP22\\PDSpeech\\";
        Path p;
        BufferedReader read;
        String line = "";
        String ime = "PDS";
        
        try{
        for(int i=0;i<40;i++){
            String inputPathR=inputPath+"Run"+(i+1)+"\\Classifier"+ime+".txt";
            p = Paths.get(inputPathR);
            read = Files.newBufferedReader(p, StandardCharsets.UTF_8);
            
            int cnt = 0;
            int first = 0;
            while((line = read.readLine())!=null){
                if(first <2){
                    first ++;
                    continue;
                }
                String tmp[] = line.split(" ");
                if(cnt == 0){
                for(int i1=1;i1<15;i1++)
                    mlp.get(i1-1).add(Double.parseDouble(tmp[i1]));
                cnt++;
                }
                else if(cnt == 1){
                    for(int i1=1;i1<15;i1++)
                    lmt.get(i1-1).add(Double.parseDouble(tmp[i1]));
                cnt++;
                }
                else if(cnt == 2){
                    for(int i1=1;i1<15;i1++)
                    nb.get(i1-1).add(Double.parseDouble(tmp[i1]));
                cnt++;
                }
                else if(cnt == 3){
                    for(int i1=1;i1<15;i1++)
                    dst.get(i1-1).add(Double.parseDouble(tmp[i1]));
                cnt++;
                }
                else if(cnt == 4){
                    for(int i1=1;i1<15;i1++)
                    log.get(i1-1).add(Double.parseDouble(tmp[i1]));
                cnt++;
                }
                else if(cnt == 5){
                   for(int i1=1;i1<15;i1++)
                    ks.get(i1-1).add(Double.parseDouble(tmp[i1]));
                cnt++; 
                }
                else if(cnt == 6){
                    for(int i1=1;i1<15;i1++)
                    rf600.get(i1-1).add(Double.parseDouble(tmp[i1]));
                cnt++;
                }
                else if(cnt == 7){
                    for(int i1=1;i1<15;i1++)
                    j48.get(i1-1).add(Double.parseDouble(tmp[i1]));
                cnt++;
                }
            }
            read.close();
            
            inputPathR=inputPath+"Run"+(i+1)+"\\Percentages"+ime+".txt";
            p = Paths.get(inputPathR);
            read = Files.newBufferedReader(p, StandardCharsets.UTF_8);
            int train = 1;
             while((line = read.readLine())!=null){
                 String tmp[] = line.split(" ");
                 if(train == 1){
                 rulePerc.add(Double.parseDouble(tmp[0].trim()));
                 suprulePerc.add(Double.parseDouble(tmp[1].trim()));
                 subgPerc.add(Double.parseDouble(tmp[2].trim()));
                 redsPerc.add(Double.parseDouble(tmp[3].trim()));
                 train = 0;
                     }
                 else{
                     rulePercTest.add(Double.parseDouble(tmp[0].trim()));
                     suprulePercTest.add(Double.parseDouble(tmp[1].trim()));
                     subgPercTest.add(Double.parseDouble(tmp[2].trim()));
                     redsPercTest.add(Double.parseDouble(tmp[3].trim()));
                 }
                     
             }
             read.close();
        }
        }
        catch(IOException e){
            e.printStackTrace();
        }
        
        try{
             FileWriter fw = new FileWriter(inputPath+"\\mlp.txt");
             for(int i=0;i<14;i++){
                 for(int j=0;j<mlp.get(i).size();j++)
                     if(j+1<mlp.get(i).size())
                         fw.write(mlp.get(i).get(j)+" ");
                     else fw.write(mlp.get(i).get(j)+"\n");           
             }
             fw.close();
             
             fw = new FileWriter(inputPath+"\\lmt.txt");
             for(int i=0;i<14;i++){
                 for(int j=0;j<lmt.get(i).size();j++)
                     if(j+1<lmt.get(i).size())
                         fw.write(lmt.get(i).get(j)+" ");
                     else fw.write(lmt.get(i).get(j)+"\n");           
             }
             fw.close();
             
             fw = new FileWriter(inputPath+"\\nb.txt");
             for(int i=0;i<14;i++){
                 for(int j=0;j<nb.get(i).size();j++)
                     if(j+1<nb.get(i).size())
                         fw.write(nb.get(i).get(j)+" ");
                     else fw.write(nb.get(i).get(j)+"\n");           
             }
             fw.close();
             
             fw = new FileWriter(inputPath+"\\dst.txt");
             for(int i=0;i<14;i++){
                 for(int j=0;j<dst.get(i).size();j++)
                     if(j+1<dst.get(i).size())
                         fw.write(dst.get(i).get(j)+" ");
                     else fw.write(dst.get(i).get(j)+"\n");           
             }
             fw.close();
             
             fw = new FileWriter(inputPath+"\\log.txt");
             for(int i=0;i<14;i++){
                 for(int j=0;j<log.get(i).size();j++)
                     if(j+1<log.get(i).size())
                         fw.write(log.get(i).get(j)+" ");
                     else fw.write(log.get(i).get(j)+"\n");           
             }
             fw.close();
             
             fw = new FileWriter(inputPath+"\\ks.txt");
             for(int i=0;i<14;i++){
                 for(int j=0;j<ks.get(i).size();j++)
                     if(j+1<ks.get(i).size())
                         fw.write(ks.get(i).get(j)+" ");
                     else fw.write(ks.get(i).get(j)+"\n");           
             }
             fw.close();
             
             fw = new FileWriter(inputPath+"\\rf600.txt");
             for(int i=0;i<14;i++){
                 for(int j=0;j<rf600.get(i).size();j++)
                     if(j+1<rf600.get(i).size())
                         fw.write(rf600.get(i).get(j)+" ");
                     else fw.write(rf600.get(i).get(j)+"\n");           
             }
             fw.close();
             
             fw = new FileWriter(inputPath+"\\j48.txt");
             for(int i=0;i<14;i++){
                 for(int j=0;j<j48.get(i).size();j++)
                     if(j+1<j48.get(i).size())
                         fw.write(j48.get(i).get(j)+" ");
                     else fw.write(j48.get(i).get(j)+"\n");           
             }
             fw.close();
             
             
           fw = new FileWriter(inputPath+"\\PercAllTrain.txt");
           
           for(int i=0;i<rulePerc.size();i++){
               if(i+1<rulePerc.size())
                   fw.write(rulePerc.get(i)+" ");
               else fw.write(rulePerc.get(i)+"\n");
           }
           
           for(int i=0;i<suprulePerc.size();i++){
               if(i+1<suprulePerc.size())
                   fw.write(suprulePerc.get(i)+" ");
               else fw.write(suprulePerc.get(i)+"\n");
           }
           
           
            for(int i=0;i<subgPerc.size();i++){
               if(i+1<subgPerc.size())
                   fw.write(subgPerc.get(i)+" ");
               else fw.write(subgPerc.get(i)+"\n");
           }
            
            for(int i=0;i<redsPerc.size();i++){
               if(i+1<redsPerc.size())
                   fw.write(redsPerc.get(i)+" ");
               else fw.write(redsPerc.get(i)+"\n");
           }
             fw.close();
             
             fw = new FileWriter(inputPath+"\\PercAllTest.txt");
           
           for(int i=0;i<rulePercTest.size();i++){
               if(i+1<rulePercTest.size())
                   fw.write(rulePercTest.get(i)+" ");
               else fw.write(rulePercTest.get(i)+"\n");
           }
           
           for(int i=0;i<suprulePercTest.size();i++){
               if(i+1<suprulePercTest.size())
                   fw.write(suprulePercTest.get(i)+" ");
               else fw.write(suprulePercTest.get(i)+"\n");
           }
           
           
            for(int i=0;i<subgPercTest.size();i++){
               if(i+1<subgPercTest.size())
                   fw.write(subgPercTest.get(i)+" ");
               else fw.write(subgPercTest.get(i)+"\n");
           }
            
            for(int i=0;i<redsPercTest.size();i++){
               if(i+1<redsPercTest.size())
                   fw.write(redsPercTest.get(i)+" ");
               else fw.write(redsPercTest.get(i)+"\n");
           }
             fw.close();
             
        }
        catch(IOException e){
            e.printStackTrace();
        }
        
    }
}
