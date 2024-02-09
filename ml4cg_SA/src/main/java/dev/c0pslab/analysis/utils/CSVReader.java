package dev.c0pslab.analysis.utils;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.LinkedHashMap;

public class CSVReader {
    public static LinkedHashMap<String, String> readProjectToJarCsv(String filePath) throws IOException {
        LinkedHashMap<String, String> jarMap = new LinkedHashMap<>();
        try (Reader reader = new FileReader(filePath)) {
            Iterable<CSVRecord> records = CSVFormat.DEFAULT.parse(reader);
            for (CSVRecord record : records) {
                if (record.size() >= 2) {
                    String project = record.get(0).trim();
                    String jarPath = record.get(1).trim();
                    jarMap.put(project, jarPath);
                }
            }
        }
        return jarMap;
    }
}
