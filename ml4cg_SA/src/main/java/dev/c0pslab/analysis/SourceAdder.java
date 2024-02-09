package dev.c0pslab.analysis;

import com.fasterxml.jackson.databind.ObjectMapper;
import dev.c0pslab.analysis.cg.CGUtils;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.apache.commons.text.similarity.JaccardSimilarity;
import org.jline.utils.Log;
import picocli.CommandLine;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

@CommandLine.Command(name="SourceAdder")
public class SourceAdder implements Runnable {

    @CommandLine.Option(names = {"-s", "--source"}, description = "source folder")
    String sourceFolder;

    // Optional
    @CommandLine.Option(names = {"-s2", "--source2"}, description = "source folder")
    Optional<String> secondSourceFolder;

    @CommandLine.Option(names = {"-d", "--dataset"}, description = "Dataset folder")
    String datasetPath;

    @CommandLine.Option(names = {"-o", "--output"}, description = "Output folder")
    String outputPath;


    @Override
    public void run() {
        try {
            addSource();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static class URIInfo {
        private String className;
        private String methodSignature;
        private String methodLineNumber;
        private String methodSource;

        public URIInfo(String className, String methodSignature, String methodLineNumber, String methodSource) {
            this.className = className;
            this.methodSignature = methodSignature;
            this.methodLineNumber = methodLineNumber;
            this.methodSource = methodSource;
        }

        public String getClassName() {
            return className;
        }

        public void setClassName(String className) {
            this.className = className;
        }

        public String getMethodSignature() {
            return methodSignature;
        }

        public void setMethodSignature(String methodSignature) {
            this.methodSignature = methodSignature;
        }

        public String getMethodLineNumber() {
            return methodLineNumber;
        }

        public void setMethodLineNumber(String methodLineNumber) {
            this.methodLineNumber = methodLineNumber;
        }

        public String getMethodSource() {
            return methodSource;
        }

        public void setMethodSource(String methodSource) {
            this.methodSource = methodSource;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            URIInfo uriInfo = (URIInfo) o;
            return new EqualsBuilder().append(className, uriInfo.className).append(methodSignature, uriInfo.methodSignature).append(methodLineNumber, uriInfo.methodLineNumber).append(methodSource, uriInfo.methodSource).isEquals();
        }

        @Override
        public int hashCode() {
            return new HashCodeBuilder(17, 37).append(className).append(methodSignature)
                    .append(methodLineNumber).append(methodSource).toHashCode();
        }
    }

    public static String removeJavaComments(String code) {
        // Removes multi-line comments
        Pattern commentPattern = Pattern.compile("/\\*.*?\\*/", Pattern.DOTALL);
        Matcher matcher = commentPattern.matcher(code);
        code = matcher.replaceAll("");

        // Removes single line comments
        commentPattern = Pattern.compile("//.*", Pattern.DOTALL);
        matcher = commentPattern.matcher(code);
        code = matcher.replaceAll("");

        return code;
    }

    public static String getBaseName(String fileName) {
        int index = fileName.lastIndexOf('.');
        if(index == -1) {
            return fileName; // if there is no "." in the file name
        } else {
            return fileName.substring(0, index);
        }
    }

    public static String extractSrcFilePath(String nodeUri) {
        String[] parts = nodeUri.split(":");
        String firstPart = parts[0];
        String fileNameWithoutExt = firstPart.split("\\.")[0];
        String fileNameWithoutDollar = fileNameWithoutExt.split("\\$")[0];
        return fileNameWithoutDollar + ".java";
    }

    public static String normalizeSrcCode(String code) {
        String[] parts = code.replace("\n", " ").split("\\s+");
        StringBuilder normalizedCode = new StringBuilder();

        for (String part : parts) {
            normalizedCode.append(part);
            normalizedCode.append(" ");
        }

        return normalizedCode.toString().trim();  // remove the trailing space
    }

    public static String extractMethodName(String nodeUri) {
        Pattern pattern = Pattern.compile(".+\\.(.+):.+");
        Matcher matcher = pattern.matcher(nodeUri);
        if (matcher.find()) {
            return matcher.group(1);
        } else {
            return "";  // or some other value indicating no match
        }
    }

    public static double getSimilarityRatio(String str1, String str2) {
        JaccardSimilarity jaccardSimilarity = new JaccardSimilarity();
        return jaccardSimilarity.apply(str1, str2);
    }

    public static boolean isWalaUriMatchMethodSig(String walaUri, String methodSig) {
        String methodSigWala = walaUri.split(":")[1];
        methodSigWala = CGUtils.simplifyMethodSignature(methodSigWala);
        String[] parts = methodSig.split("\\(");
        String methodName = parts[0];
        methodSig = "(" + parts[1];
        return extractMethodName(walaUri).equals(methodName) &&
                getSimilarityRatio(methodSig, methodSigWala) > 0.50;
    }

    public static void main(String[] args) throws IOException {
        System.exit(new CommandLine(new SourceAdder()).execute(args));
    }

    private void addSource() throws IOException {
        List<Path> projectsSrc = new ArrayList<>();
        BiConsumer<List<Path>, String> getSourceJSONs = (projects, folder) -> {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(Paths.get(folder))) {
                for (Path path : stream) {
                    if (!Files.isDirectory(path) && path.toString().endsWith(".json")) {
                        projects.add(path);
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        };
        getSourceJSONs.accept(projectsSrc, sourceFolder);
        secondSourceFolder.ifPresent(folder -> getSourceJSONs.accept(projectsSrc, folder));
        System.out.println("h");

        Map<String, List<URIInfo>> filesToUriInfo = new HashMap<>();
        ObjectMapper mapper = new ObjectMapper();
        Consumer<Path> processJSONFile = (path) -> {
            try {
                List<Map<String, Object>> data = mapper.readValue(Files.newBufferedReader(path),
                        mapper.getTypeFactory().constructCollectionType(List.class, Map.class));

                for (Map<String, Object> record : data) {
                    final String className = (String) record.get("className");
                    final String methodSignature = removeJavaComments((String) record.get("methodSignature"));
                    final String methodLineNumbers = (String) record.get("methodLineNumbers");
                    final String methodSource = (String) record.get("methodSource");

                    URIInfo uriInfo = new URIInfo(className, methodSignature, methodLineNumbers, methodSource);
                    var pathToRemove = ((String) record.get("sourceFilePath")).indexOf('/');
                    String sourceFilePath = ((String) record.get("sourceFilePath")).substring(pathToRemove + 1);

                    if (!filesToUriInfo.containsKey(sourceFilePath)) {
                        filesToUriInfo.put(sourceFilePath, new ArrayList<>());
                    }

                    filesToUriInfo.get(sourceFilePath).add(uriInfo);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        };
        projectsSrc.forEach(processJSONFile);

        String [] headers = {"wiretap", "wala-cge-0cfa-noreflect-intf-trans", "method", "offset",
                "target", "method_ln", "target_ln", "method_src", "target_src",
                "program_name"};
        Reader reader = new FileReader(datasetPath);
        var csvFormat = CSVFormat.DEFAULT.builder().setHeader(headers).setSkipHeaderRecord(true).build();
        var records = csvFormat.parse(reader);
        var numRepairedRecords = new AtomicInteger(0);
        List<Map<String, String>> repairedRecords = Collections.synchronizedList(new ArrayList<>());
        records.stream().parallel().forEach( (csvRecord) -> {
            final Map<String, String> recordMap = new HashMap<>(csvRecord.toMap());

            final String methodURI = csvRecord.get("method");
            final String targetURI = csvRecord.get("target");
            final String methodFileName = extractSrcFilePath(methodURI);
            final String targetFileName = extractSrcFilePath(targetURI);
            final String methodLn = csvRecord.get("method_ln");
            final String targetLn = csvRecord.get("target_ln");
            if (filesToUriInfo.containsKey(methodFileName)) {
                final var foundMethodSource = findAddSource(filesToUriInfo, methodURI, methodFileName, methodLn);
                if (foundMethodSource != null) {
                    recordMap.put("method_src", foundMethodSource);
                    numRepairedRecords.incrementAndGet();
                }

            if (filesToUriInfo.containsKey(targetFileName)) {
                final var foundTargetSource = findAddSource(filesToUriInfo, targetURI, targetFileName, targetLn);
                if (foundTargetSource != null) {
                    recordMap.put("target_src", foundTargetSource);
                    numRepairedRecords.incrementAndGet();
                }
                }
            }
            repairedRecords.add(recordMap);
        });
        Log.info("Repaired " + numRepairedRecords + " samples/edges with source code");
        FileWriter out = new FileWriter(outputPath);
        CSVPrinter printer = CSVFormat.DEFAULT.builder().setHeader(headers).build().print(out);
        for (Map<String, String> recordMap : repairedRecords) {
            printer.printRecord(Arrays.stream(headers)
                    .map(recordMap::get)
                    .collect(Collectors.toList()));
        }
        out.close();
    }

    private static String findAddSource(Map<String, List<URIInfo>> filesToUriInfo, String methodURI,
                                        String methodFileName, String methodLn) {
        for (var u : filesToUriInfo.get(methodFileName)) {
            // Not effective if line info from bytecode is incorrect
//            var methodLnSplit = methodLn.split(" ");
//            if (!methodLnSplit[0].equals("-1") && !methodLnSplit[1].equals("-1")) {
//                var lnURI = u.getMethodLineNumber().split(" ");
//                if (Integer.parseInt(methodLnSplit[0]) <= Integer.parseInt(lnURI[0]) &&
//                        Integer.parseInt(methodLnSplit[1]) <= Integer.parseInt(lnURI[1])) {
//                    return normalizeSrcCode(u.getMethodSource());
//                }
//            }

            if (isWalaUriMatchMethodSig(methodURI, u.getMethodSignature())) {
                return normalizeSrcCode(u.getMethodSource());
            }
        }
        return null;
    }
}