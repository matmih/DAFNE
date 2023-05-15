/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package datasetcreator;

/**
 *
 * @author Matej
 */
public class NumberFormat {
    public static void main(String args[]){
     //  String num = "9.999951E−1";
        String num = "9.999951E-1";
        double d = Double.valueOf(num);
        System.out.println(d);
    }
}
