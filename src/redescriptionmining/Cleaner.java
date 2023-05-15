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
public class Cleaner {
    public void clean(String input){
        BufferedReader read = null;
        FileWriter fw = null;
        ArrayList<String> lines = new ArrayList<>();
        
        try{
            Path p = Paths.get(input);
            read = Files.newBufferedReader(p,StandardCharsets.UTF_8);
            String line = "";
            while((line = read.readLine())!=null){
                line = line.replaceAll("−", "-");
                lines.add(line);
            }
            read.close();
            
            fw = new FileWriter(input);
            
            for(int i=0;i<lines.size();i++)
                fw.write(lines.get(i)+"\n");
            fw.close();
        }
        catch(IOException e){
            e.printStackTrace();
        }
    }
}
