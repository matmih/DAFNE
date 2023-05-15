/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package redescriptionmining;

import java.io.File;
import java.util.ArrayList;
import static junit.framework.Assert.assertTrue;

/**
 *
 * @author matej
 */
public class UnitTestsCLUS_RM {
    static public void main(String [] args){
        
          ApplicationSettings appset=new ApplicationSettings();
        appset.readSettings(new File(args[0]));
             appset.readPreference(); 
        CLUS_RMRun crun = new CLUS_RMRun();
        ArrayList<RedescriptionSet> res =crun.run(appset);
        
         DataSetCreator datJ=new DataSetCreator(appset.viewInputPaths, appset.outFolderPath,appset);
        Mappings fid=new Mappings();
        
        if(appset.system.equals("windows"))
            fid.createIndex(appset.outFolderPath+"\\Jinput.arff");
        else
            fid.createIndex(appset.outFolderPath+"/Jinput.arff");
        
       assertTrue(res.size()>0);
        
       if(res.size()==1){
           RedescriptionSet rs = res.get(0);
           
           for(Redescription r:rs.redescriptions){
               assertTrue(r.JS>=appset.minJS);
               assertTrue(r.pVal<=appset.maxPval);
               assertTrue(r.elements.size()>=appset.minSupport && r.elements.size()<=appset.maxSupport);
               assertTrue(r.elements.size()/((double)r.elementsUnion.size()) == r.JS);
               assertTrue(r.pVal == r.computePValS(datJ, fid));
           }
           
       }
        for(RedescriptionSet rs:res){
            
        }

//int a = 4, b=5;
        
       // assertTrue(a>b);
        
    }
}
