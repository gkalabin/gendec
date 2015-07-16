package com.github.gkalabin.gendec;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.util.stream.Collectors.*;

/**
 * @author grigory.kalabin@gmail.com
 **/
public class GenderDetector {
  private static final int MOST_FREQUENT_TRIGRAMS_COUNT = 40;
  private static final int MAX_TRAIN_ITERATIONS = 1000;
  // name or surname -> probability to be a man with such name or surname
  private static Map<String, Double> menProbability;
  // collection with most frequent trigrams across names and surnames
  private static List<String> mostFrequentTrigrams;
  // optimized parameters for regression
  private static double[] theta;

  public static void main(String[] args) throws IOException {
    if (args.length != 2 || !(new File(args[0])).exists() || !(new File(args[1])).exists()) { throw new IllegalArgumentException("Two args: train file and test file"); }
    String train = args[0];
    String test = args[1];

    List<String> lines = Files.readAllLines(Paths.get(train));
    List<String> names = lines.stream().map(GenderDetector::getName).collect(Collectors.toList());
    List<String> surnames = lines.stream().map(GenderDetector::getSurname).collect(Collectors.toList());

    // there are many opportunities for optimization in trigrams processing: solution is very straightforward
    List<String> trigrams = names.stream()
      .map(GenderDetector::trigrams)
      .flatMap(List::stream)
      .collect(Collectors.toList());
    trigrams.addAll(surnames.stream()
      .map(GenderDetector::trigrams)
      .flatMap(List::stream)
      .collect(Collectors.toList())
    );
    Map<String, Long> trigramCounts = trigrams.stream().collect(groupingBy(Function.identity(), counting()));
    List<Map.Entry<String, Long>> entries = new ArrayList<>(trigramCounts.entrySet());
    Collections.sort(entries, Map.Entry.comparingByValue(Long::compare));
    mostFrequentTrigrams = entries.stream()
      .map(Map.Entry::getKey)
      .collect(Collectors.toList())
      .subList(entries.size() - MOST_FREQUENT_TRIGRAMS_COUNT, entries.size());
    Collections.sort(mostFrequentTrigrams);

    // code reuse suffer
    List<String> menNamesAndSurnames = lines.stream()
      .filter(GenderDetector::isMale)
      .map(l -> l.substring(0, l.length() - 2))
      .flatMap(l -> Arrays.stream(l.split(" ")))
      .collect(toList());
    Map<String, Long> menNameSurnameFrequency = menNamesAndSurnames.stream().collect(groupingBy(Function.identity(), counting()));
    List<String> womenNamesAndSurnames = lines.stream()
      .filter(GenderDetector::isFemale)
      .map(l -> l.substring(0, l.length() - 2))
      .flatMap(l -> Arrays.stream(l.split(" ")))
      .collect(toList());
    Map<String, Long> womenNameSurnameFrequency = womenNamesAndSurnames.stream().collect(groupingBy(Function.identity(), counting()));
    Set<String> namesAndSurnames = new HashSet<>(menNamesAndSurnames);
    namesAndSurnames.addAll(womenNamesAndSurnames);
    menProbability = namesAndSurnames.stream()
      .collect(toMap(
        Function.identity(),
        nameOrSurname -> (double) menNameSurnameFrequency.getOrDefault(nameOrSurname, 0L) /
          (menNameSurnameFrequency.getOrDefault(nameOrSurname, 0L) + womenNameSurnameFrequency.getOrDefault(nameOrSurname, 0L))
      ));

    //
    double[][] features = new double[lines.size()][];
    double[] target = new double[lines.size()];
    for (int i = 0; i < lines.size(); i++) {
      features[i] = getFeatures(lines.get(i));
      target[i] = isMale(lines.get(i)) ? 1d : 0d;
    }
    int steps = 0;
    double[] oldTheta = new double[features[0].length];
    Arrays.fill(oldTheta, .1d);
    double[] newTheta = new double[features[0].length];
    double step = .01d;
    double epsilon = .007d;
    double oldValue = 0;
    double newValue = epsilon + 1;
    while (steps < MAX_TRAIN_ITERATIONS && Math.abs(oldValue - newValue) > epsilon) {
      double[] gradient = getValueGradient(oldTheta, features, target);
      for (int i=0; i<newTheta.length; i++) {
        double tmp = newTheta[i];
        newTheta[i] = oldTheta[i] + step * gradient[i];
        oldTheta[i] = tmp;
      }
      oldValue = getValue(oldTheta, features, target);
      newValue = getValue(newTheta, features, target);
      steps++;
    }

    // predict
    theta = newTheta;
    System.out.println(" === test ===");
    predictAndPrintMeasures(Files.readAllLines(Paths.get(test)));
    System.out.println(" === train ===");
    predictAndPrintMeasures(Files.readAllLines(Paths.get(train)));
  }

  private static void predictAndPrintMeasures(List<String> lines) {
    int tp = 0, fp = 0, tn =0, fn = 0;
    for (String line : lines) {
      double[] lineFeatures = getFeatures(line);
      double maleProbability = h(theta, lineFeatures);
      boolean predictedMale = maleProbability > 0.5d;
      boolean trueMale = isMale(line);
      if (predictedMale && trueMale) { tp++; }
      if (predictedMale && !trueMale) { fp++; }
      if (!predictedMale && trueMale) { fn++; }
      if (!predictedMale && !trueMale) { tn++; }
      if (trueMale != predictedMale) {
        System.out.printf("%s %.3f\n", line, maleProbability);
      }
    }
    System.out.printf("%d\t%d\n%d\t%d\n", tp, fp, fn, tn);
  }

  private static double[] getFeatures(String line) {
    String name = getName(line);
    String surname = getSurname(line);
    double[] features = new double[MOST_FREQUENT_TRIGRAMS_COUNT + 2];
    for (int i=0; i<mostFrequentTrigrams.size(); i++) {
      features[i] = testTrigram(name, mostFrequentTrigrams.get(i)) ? 1d : 0d;
      features[i] += testTrigram(surname, mostFrequentTrigrams.get(i)) ? 1d : 0d;
    }
    features[features.length - 2] = menProbability.getOrDefault(name, 0.5d);
    features[features.length - 1] = menProbability.getOrDefault(surname, 0.5d);
    return features;
  }

  private static String getName(String line) { return line.split(" ")[0]; }
  private static String getSurname(String line) { return line.split(" ")[1]; }
  private static boolean isMale(String line) { return line.split(" ")[2].equals("M"); }
  private static boolean isFemale(String line) { return line.split(" ")[2].equals("F"); }

  private static List<String> trigrams(String nameOrSurname) {
    List<String> trigrams = new ArrayList<>();
    for (int i=-2; i<nameOrSurname.length(); i++) {
      String prefix = "";
      String suffix = "";
      if (i == -2) { prefix = ".."; } else if (i == -1) { prefix = "."; } else
      if (i == nameOrSurname.length() - 2) { suffix = "."; } else if (i == nameOrSurname.length() - 1) { suffix = ".."; }
      trigrams.add(prefix + nameOrSurname.substring(Math.max(i, 0), Math.min(i + 3, nameOrSurname.length())) + suffix);
    }
    return trigrams;
  }

  private static boolean testTrigram(String line, String trigram) {
    if (trigram.startsWith("..")) { return line.startsWith(trigram.substring(2,3)); } else
    if (trigram.startsWith("."))  { return line.startsWith(trigram.substring(1,3)); } else
    if (trigram.endsWith(".."))  { return line.endsWith(trigram.substring(0,1)); } else
    if (trigram.endsWith("."))   { return line.endsWith(trigram.substring(0,2)); } else
      return line.contains(trigram);
  }

  private static double h(double[] theta, double[] x) {
    double sum = 0;
    for (int i=0; i< theta.length; i++) {
      sum += theta[i] * x[i];
    }
    return 1d/(1d + Math.pow(Math.E, -sum));
  }

  private static final double LAMBDA = .1d;
  private static double getValue(double[] theta, double[][] features, double[] target) {
    double value = 0d;
    for (int line = 0; line < target.length; line++) {
      double logitValue = h(theta, features[line]);
      value += target[line] * Math.log(logitValue) + (1 - target[line]) * Math.log(1 - logitValue);
    }
    for (int i=0; i< theta.length; i++) {
      value -= theta[i]*theta[i] * LAMBDA;
    }
    return value;
  }

  private static double[] getValueGradient(double[] theta, double[][] features, double[] target) {
    double[] gradient = new double[features[0].length];
    for (int j=0; j< gradient.length; j++) {
      gradient[j] = -2 * LAMBDA * theta[j];
      // gradient[j] = 0;
      for (int i =0; i<target.length; i++) {
        gradient[j] += (- h(theta, features[i]) + target[i]) * features[i][j];
      }
      // gradient[j] = -gradient[j];
    }
    return gradient;
  }
}
