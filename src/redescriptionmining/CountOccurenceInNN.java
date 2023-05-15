/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package redescriptionmining;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 *
 * @author Matej
 */
public class CountOccurenceInNN {
    public static void main(String args[]){
        String inputPath = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\Eksperimenti konstrukcija značajki\\NFMCP22\\Sports Articles\\";
        Path p;
        BufferedReader read;
        String line = "";
        String ime = "SportsArticles";
        
        try{
             int nsg = 0, nsr = 0, ndr = 0, nrd=0;
        for(int i=0;i<40;i++){
            int foundSg = 0, foundSR = 0, foundDR = 0, foundRD = 0;
            String inputPathR=inputPath+"Run"+(i+1)+"\\ FeatureImportanceNonRedundant"+ime+" .txt";
            if(i == 0 || i ==1)
                inputPathR=inputPath+"Run"+(i+1)+"\\FeatureImportanceNonRedundant"+ime+".txt";
            if(i>1)
                 inputPathR=inputPath+"Run"+(i+1)+"\\FeatureImportanceNonRedundant"+ime+" .txt";
            p = Paths.get(inputPathR);
            read = Files.newBufferedReader(p, StandardCharsets.UTF_8);
            
            int cnt = 0;
            int first = 0;
           
            while((line = read.readLine())!=null){
                if(first <1){
                    first ++;
                    continue;
                }
                
                String tmp[] = line.split("\t");
                String at = tmp[1].trim();
                
                boolean result = at.matches("\"r\\d+\"");
                
                if(result && foundDR == 0){ ndr++; foundDR=1; continue;}
                
                result = at.matches("\"rs\\d+\"");
                
                if(result && foundSR == 0){ nsr++; foundSR = 1; continue;}
                
                result = at.matches("\"S\\d+\"");
                if(result && foundSg == 0){ nsg++; foundSg = 1; continue;}
                
                result = at.matches("\"R\\d+\"");
                if(result && foundRD == 0){ nrd++; foundRD = 1; continue;}
                
            }
                read.close();
        }
        System.out.println(nsg+" "+nsr+" "+ndr+" "+nrd);
        }
        catch(IOException e){
            e.printStackTrace();
        }
    }
}
