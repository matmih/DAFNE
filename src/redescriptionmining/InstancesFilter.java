/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package redescriptionmining;

/**
 *
 * @author matej
 */
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.Center;
import weka.filters.unsupervised.attribute.RemoveUseless;

/**
* Performs data pre-processing by applying filters to a set of Weka instances.
*/
public class InstancesFilter {

    /**
     * Makes numeric attributes have zero mean
     *
     * @throws Exception
     *             If filter could not be applied
     */
    public void centerFilter() throws Exception {

        // Might employ filtered classifier for production
        Center ct = new Center();
        ct.setInputFormat(_instances);
        _instances = Filter.useFilter(_instances, ct);
    }

    /**
     * Generates mapping from attribute name to an integer value. Used by
     * removeNameFilter
     */
    private void generateAttributeMap() {

        _attributeMap = new HashMap<String, Integer>();
        int numAttributes = _instances.numAttributes();
        for (int i = 0; i < numAttributes; i++) {
            _attributeMap.put(_instances.attribute(i).name().trim(), i);
        }
    }

    /**
     * @return The filtered instances
     */
    public Instances getFilteredInstances() {
        return _instances;
    }

    /**
     * Applies a filter to remove supplied attribute names from the set of
     * instances
     *
     * @param names
     *            The name(s) of the attribute(s) to remove
     * @throws Exception
     *             If filter could not be applied
     */
    public void removeNameFilter(String[] names) throws Exception {

        // Might employ filtered classifier for production
        MultiFilter mf = new MultiFilter();
        String[] options = new String[names.length * 2];
        for (int i = 0; i < options.length; i++) {
            if (i % 2 == 0) {
                options[i] = "-F";
            } else {
                options[i] = "weka.filters.unsupervised.attribute.Remove -R " + _attributeMap.get(names[i / 2]);
            }
        }
        mf.setOptions(options);
        mf.setInputFormat(_instances);
        _instances = Filter.useFilter(_instances, mf);
    }

    /**
     * Applies a filter to remove stratified folds from the set of instances
     *
     * @param fold
     *            The fold number to pick for remove
     * @param numFolds
     *            The number of folds for a stratified cross-validation
     * @param invert
     *            Flag indicating whether to remove this fold or all others
     * @throws Exception
     *             If filter could not be applied
     */
    public void removeStratifiedFoldsFilter(Integer fold, Integer numFolds, boolean invert) throws Exception {

        StratifiedRemoveFolds srf = new StratifiedRemoveFolds();
        String[] options;
        if (invert) {
            options = new String[6];
            options[0] = "-S";
            options[1] = "-9";
            options[2] = "-N";
            options[3] = numFolds.toString();
            options[4] = "-F";
            options[5] = fold.toString();
        } else {
            options = new String[7];
            options[0] = "-S";
            options[1] = "-9";
            options[2] = "-V";
            options[3] = "-N";
            options[4] = numFolds.toString();
            options[5] = "-F";
            options[6] = fold.toString();
        }
        srf.setOptions(options);
        srf.setInputFormat(_instances);
        _instances = Filter.useFilter(_instances, srf);
    }

    /**
     * Applies a filter to remove supplied attribute types
     *
     * @param types
     *            The name(s) of the attribute type(s) to remove
     * @throws Exception
     *             Ifs filter could not be applied
     */
    public void removeTypeFilter(String[] types) throws Exception {

        // Might employ filtered classifier for production
        MultiFilter mf = new MultiFilter();
        String[] options = new String[types.length * 2];
        for (int i = 0; i < options.length; i++) {
            if (i % 2 == 0) {
                options[i] = "-F";
            } else {
                options[i] = "weka.filters.unsupervised.attribute.RemoveType -T " + types[i / 2];
            }
        }
        mf.setOptions(options);
        mf.setInputFormat(_instances);
        _instances = Filter.useFilter(_instances, mf);
    }

    /**
     * Applies a filter to remove useless attributes with a variance greater
     * than the specified value
     *
     * @param variance
     *            The maximum variance for the attribute
     * @throws Exception
     *             If filter could not be applied
     */
    public void removeUselessFilter(String variance) throws Exception {
  
        // Might employ filtered classifier for production
        RemoveUseless ru = new RemoveUseless();
        String[] options = new String[2];
        options[0] = "-M";
        options[1] = variance;
        ru.setOptions(options);
        ru.setInputFormat(_instances);
        _instances = Filter.useFilter(_instances, ru);
    }

    /**
     * @param instances
     *            The instances to set
     */
    private void setInstances(Instances instances) {
        _instances = instances;
    }

    /**
     * contains a mapping from attribute name to location
     */
    private Map<String, Integer> _attributeMap;

    /**
     * object to hold filtered instances
     */
    private Instances _instances;

    /**
     * a handle to the logger
     */
    private Logger _logger;

    /**
     * Class constructor
     *
     * @param instances
     *            The set of instances to filter
     */
    public InstancesFilter(Instances instances) {
        setInstances(new Instances(instances));       
        generateAttributeMap();
    }

}