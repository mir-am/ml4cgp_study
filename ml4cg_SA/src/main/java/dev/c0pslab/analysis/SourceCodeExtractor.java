package dev.c0pslab.analysis;

import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.github.javaparser.ParseProblemException;
import com.github.javaparser.Range;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.DirectoryFileFilter;
import org.apache.commons.io.filefilter.SuffixFileFilter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;

import java.io.File;
import java.io.IOException;
import java.nio.charset.MalformedInputException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class SourceCodeExtractor implements Runnable {

    static class Method {

        public Method() {

        }
        public Method(String sourceFilePath, String className, String methodSignature,
                      String methodLineNumbers, String methodSource) {
            this.sourceFilePath = sourceFilePath;
            this.className = className;
            this.methodSignature = methodSignature;
            this.methodLineNumbers = methodLineNumbers;
            this.methodSource = methodSource;
        }

        public String sourceFilePath;
        public String className;
        public String methodSignature;
        public String methodLineNumbers;
        public String methodSource;

    }

    private static final Logger LOG = LoggerFactory.getLogger(SourceCodeExtractor.class);

    @CommandLine.Option(names = {"-d", "--dataset"}, description = "Dataset folder")
    String datasetPath;

    @CommandLine.Option(names = {"-o", "--output"}, description = "Output folder")
    File outputFolder;

    @Override
    public void run() {
        StaticJavaParser.getParserConfiguration().setAttributeComments(false);
        StaticJavaParser.getParserConfiguration().setCharacterEncoding(StandardCharsets.ISO_8859_1);
        try {
            if (datasetPath != null && outputFolder != null) {
                extractSourceCodeFromDataset();
            } else {
                throw new RuntimeException("Provide dataset and output folders path!");
            }

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void extractSourceCodeFromDataset() throws IOException {
        List<File> sourceFolders = findSourceFolders(datasetPath);
        sourceFolders.parallelStream().forEach(folder -> {
            final String projectFolder = folder.getName();
            File projectJsonFile = Paths.get(outputFolder.getAbsolutePath(), projectFolder + ".json").toFile();
            if (projectJsonFile.exists()) {
                LOG.info("Project " + projectFolder + " already processed!");
                return;
            }
            Collection<File> projectSourceFiles = FileUtils.listFiles(folder,
                    new SuffixFileFilter(new String[] {"java"}), DirectoryFileFilter.DIRECTORY);
            ArrayList<Method> projectMethods = new ArrayList<>();
            ObjectMapper mapper = new ObjectMapper();
            ObjectWriter writer = mapper.writer(new DefaultPrettyPrinter());
            projectSourceFiles.forEach(f -> {
                try {
                    projectMethods.addAll(parseSourceCode(f.getAbsolutePath(), folder.getParentFile().getAbsolutePath()));
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            });
            try {
                if (projectMethods.size() != 0) {
                    writer.writeValue(projectJsonFile, projectMethods);
                }
                LOG.info("Processed the project " + projectFolder);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    public static List<File> findSourceFolders(String directoryPath) {
        List<File> sourceFolders = new ArrayList<>();
        File directory = new File(directoryPath);
        if (directory.exists() && directory.isDirectory()) {
            File[] files = directory.listFiles();
            for (File file : files) {
                if (file.isDirectory() && file.getName().endsWith("-sources")) {
                    sourceFolders.add(file);
                } else if (file.isDirectory()) {
                    sourceFolders.addAll(findSourceFolders(file.getAbsolutePath()));
                }
            }
        }
        return sourceFolders;
    }

    public static void main(String[] args) {
        System.exit(new CommandLine(new SourceCodeExtractor()).execute(args));
    }

    private static ArrayList<Method> parseSourceCode(String filePath, String projectPath) throws IOException {
        // TODO: Ignore package-info files
        ArrayList<Method> fileMethods = new ArrayList<>();
        try {
            final String fileSource = Files.readString(Paths.get(filePath), StandardCharsets.ISO_8859_1);
            try {
                CompilationUnit cu = StaticJavaParser.parse(fileSource);
                cu.findAll(ClassOrInterfaceDeclaration.class).forEach(c -> {
                    final Optional<String> classFQN = c.getFullyQualifiedName();
                    c.findAll(MethodDeclaration.class).forEach(m -> {
                        final String relativeSourcePath = Paths.get(projectPath).relativize(Paths.get(filePath)).toString();
                        final String callableDeclaringClass = c.getName().asString();
                        final String callableSignature = m.getSignature().toString() + m.getType();
                        final String callableSource = m.toString();
                        final Range callablePosition = m.getRange().get();
                        final String callableLineNumbers = callablePosition.begin.line + " " + callablePosition.end.line;
                        fileMethods.add(new Method(relativeSourcePath, callableDeclaringClass, callableSignature,
                                callableLineNumbers, callableSource));
                    });
                });
                for (var m : fileMethods) {
                    if (m.className.contains("ViewMapIterator")) {
                        System.out.println("s");
                    }
                }
                return fileMethods;
            } catch (ParseProblemException e) {
                LOG.error("Could not parse the file: " + filePath);
//                        e.printStackTrace();
            }
        } catch (MalformedInputException e) {
            System.out.println("What?");
        }
        return fileMethods;
    }
}
