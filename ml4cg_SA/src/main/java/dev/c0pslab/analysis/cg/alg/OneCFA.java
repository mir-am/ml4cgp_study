package dev.c0pslab.analysis.cg.alg;

import com.ibm.wala.classLoader.Language;
import com.ibm.wala.ipa.callgraph.AnalysisCacheImpl;
import com.ibm.wala.ipa.callgraph.AnalysisOptions;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.CallGraphBuilderCancelException;
import com.ibm.wala.ipa.callgraph.impl.Util;
import com.ibm.wala.ipa.callgraph.propagation.SSAPropagationCallGraphBuilder;
import com.ibm.wala.ipa.cha.ClassHierarchy;
import dev.c0pslab.analysis.cg.*;
import dev.c0pslab.analysis.utils.CSVReader;
import dev.c0pslab.analysis.utils.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.*;

/*
    Creates a call graph based on the 1-CFA approach, which is context-sensitive.
 */
public class OneCFA {
    private static final Logger LOG = LoggerFactory.getLogger(OneCFA.class);
    private static final int N = 1;

    static public void buildCG(final String inputJars, final String outputPath) throws IOException {
        var trainProgramsJars = CSVReader.readProjectToJarCsv(inputJars);
//        var programsSource2TargetsCandidates = new ConcurrentHashMap<String, CallGraph>();
        final var filteredTrainProgramsJars = trainProgramsJars;
//        final var filteredTrainProgramsJars = trainProgramsJars.entrySet().stream()
//                .filter(entry -> {
//                    final var programJarFile = Paths.get(entry.getValue());
//                    final var programJarName = entry.getKey();
//                    var programOutputCGFile = Paths.get(outputPath, programJarName,
//                            programJarFile.getFileName().toString().replace(".jar", "") +  "_1cfa.csv");
//                    return !Files.exists(programOutputCGFile);
//                })
//                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
//        var customThreadPool = new ForkJoinPool(GlobalConstants.maxNumberOfThreads);
//        try {
//            customThreadPool.submit(() -> filteredTrainProgramsJars.entrySet().parallelStream().forEach((programJar) -> {
                for (var programJar : filteredTrainProgramsJars.entrySet()) {
                    final var programJarFile = Paths.get(programJar.getValue());
                    final var programJarFileDeps = CGUtils.getProgramDependencies(programJarFile);
                    final var programJarName = programJar.getKey();
                    // programJarName.contains("/") ? programJarName.replace("/", "_") :
                    final var programOutputCGFolder = Paths.get(outputPath, programJarName);
                    final var programOutputCGFile = Paths.get(programOutputCGFolder.toString(), programJarFile.getFileName().toString().replace(".jar", ""));

                    final var cha = new CHA();
                    final ClassHierarchy chaMap;
                    try {
                        chaMap = cha.construct(programJarFile.toString(), programJarFileDeps,
                                CGUtils.createWalaExclusionFile().toFile());
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                    final var entryPoints = new EntryPointsGenerator(chaMap).getEntryPoints();
                    final var opts = new AnalysisOptions(cha.getCHAScope(), entryPoints);
                    // No reflection
                    opts.setReflectionOptions(AnalysisOptions.ReflectionOptions.NONE);
                    final var cache = new AnalysisCacheImpl();
                    LOG.info("Building a call graph for " + programJarName);
                    final var startTime = System.nanoTime();
                    // Util.makeNCFABuilder(N, opts, cache, chaMap);
                    final SSAPropagationCallGraphBuilder cgBuilder = Util.makeZeroOneCFABuilder(Language.JAVA, opts, cache, chaMap);
                    Callable<Void> task = () -> {
                        final CallGraph cg;
                        try {
                            cg = cgBuilder.makeCallGraph(opts, null);
                            double durationInSeconds = (System.nanoTime() - startTime) / 1_000_000_000.0;
                            LOG.info("Generated a call graph for " + programJarFile.getFileName() + " in " + durationInSeconds + " seconds");
                            Files.createDirectories(programOutputCGFolder);
                            IOUtils.writeWalaCGToFile(cg, programOutputCGFile.toString() + "_1cfa");
                        } catch (CallGraphBuilderCancelException | IOException e) {
                            LOG.error("Failed to generate a call graph for " + programJarFile.getFileName());
                            e.printStackTrace();
                        }
                        return null;
                    };
                    ExecutorService executor = Executors.newSingleThreadExecutor();
                    Future<Void> future = executor.submit(task);

                    try {
                        future.get(GlobalConstants.maxHoursToGenerateCG, TimeUnit.HOURS);
                    } catch (TimeoutException e) {
                        LOG.error("Failed to generate a call graph for " + programJarFile.getFileName() +" in the specified time");
                        future.cancel(true);
                    } catch (InterruptedException | ExecutionException e) {
                        LOG.error("Failed to generate a call graph for " + programJarFile.getFileName());
                    } finally {
                        executor.shutdownNow();
                    }
                }
    }
}
