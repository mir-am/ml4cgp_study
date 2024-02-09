package dev.c0pslab.analysis;

import dev.c0pslab.analysis.cg.GlobalConstants;
import dev.c0pslab.analysis.cg.alg.*;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;

import java.io.IOException;

public class CGGenRunner implements Runnable {
    private static final Logger LOG = LoggerFactory.getLogger(CGGenRunner.class);

    @CommandLine.Option(names = {"-o", "--output"}, description = "An output path with a file name to save source" +
            " to target candidates")
    String outputJSONFile;

    @CommandLine.Option(names = {"-j", "--jars"}, description = "A CSV file containing list of project names and " +
            "their respective JAR files to be analyzed")
    String inputJars;

    // CG algorithms
    @CommandLine.Option(names = {"-0cfa", "--zerocfa"}, defaultValue = "false" ,description = "Build a CG with the 0-CFA algorithm")
    boolean useZeroCFA;

    @CommandLine.Option(names = {"-1cfa", "--onecfa"}, defaultValue = "false" ,description = "Build a CG with the 1-CFA algorithm")
    boolean useOneCFA;

    @CommandLine.Option(names = {"-p", "--parallel"}, defaultValue = "false" ,description = "Build a CG with the RTA algorithm")
    boolean useParallelism;

    @CommandLine.Option(names = {"-t", "--threads"}, defaultValue = "1" ,description = "Number of threads for building call graphs",
            type = int.class)
    int numberOfThreads;

    @Override
    public void run() {
        GlobalConstants.useParallelism = useParallelism;
        GlobalConstants.maxNumberOfThreads = numberOfThreads;
        LOG.info("Max Heap size: " + FileUtils.byteCountToDisplaySize(Runtime.getRuntime().maxMemory()));
        LOG.info("Number of threads to use: " + GlobalConstants.maxNumberOfThreads);
        final var startTime = System.nanoTime();
        try {
            if (useZeroCFA) {
                LOG.info("Using the 0-CFA approach");
               buildCGsWithZeroCFA();
            } else if (useOneCFA) {
                LOG.info("Using the 1-CFA approach");
                buildCGsWithOneCFA();
            } else {
                throw new RuntimeException("Choose a CG algorithm!");
            }
            final var elapsedTimeInSeconds = (double)(System.nanoTime() - startTime) / 1_000_000_000.0;
            LOG.info("Elapsed Time: " + String.format("%.02f", elapsedTimeInSeconds) + " seconds");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) {
        System.exit(new CommandLine(new CGGenRunner()).execute(args));
    }

    private void buildCGsWithZeroCFA() throws IOException {
        ZeroCFA.buildCG(inputJars, outputJSONFile);
    }

    private void buildCGsWithOneCFA() throws IOException {
        OneCFA.buildCG(inputJars, outputJSONFile);
    }
}
