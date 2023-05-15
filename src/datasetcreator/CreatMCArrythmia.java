/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package datasetcreator;

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
public class CreatMCArrythmia {
    public static void main(String args[]){
        String input = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\DS2019\\Arrhythmia\\arrhythmia.arff";
        String output = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\DS2019\\Arrhythmia\\arrhythmiaMC.arff";
        String data = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\DS2019\\Arrhythmia\\arrhythmia.data";
        
        ArrayList<Integer> classes = new ArrayList<>();
        
        try{
            Path p = Paths.get(data);
            BufferedReader read = Files.newBufferedReader(p, StandardCharsets.UTF_8);
            String line = "";
            while((line = read.readLine())!=null){
                String tmp[] = line.split(",");
                classes.add(Integer.parseInt(tmp[tmp.length-1]));
            }
            read.close();
            
            p = Paths.get(input);
            read = Files.newBufferedReader(p, StandardCharsets.UTF_8);
            FileWriter fw = new FileWriter(output);
            int data1 = 0, j = 0;
             while((line = read.readLine())!=null){
                 if(data1==0){
                     fw.write(line+"\n");
                     if(line.toLowerCase().contains("@data"))
                         data1 = 1;
                     continue;
                 }
                 if(line.equals("")){ 
                     fw.write("\n");
                     continue;
                 }
                String tmp[] = line.split(",");
                String nl = "";
                for(int i=0;i<tmp.length-1;i++)
                    nl+=tmp[i]+",";
                nl+=classes.get(j)+"\n";
                j++;
                fw.write(nl);
            }
            read.close();
            fw.close();
        }
        catch(IOException e){
            e.printStackTrace();
        }
        
    }
}
