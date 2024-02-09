package dev.c0pslab.analysis.utils;

import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.types.ClassLoaderReference;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.Selector;
import com.ibm.wala.types.TypeName;
import dev.c0pslab.analysis.cg.CGUtils;
import dev.c0pslab.analysis.cg.GlobalConstants;
import dev.c0pslab.analysis.cg.ds.CallGraphNode;
import dev.c0pslab.analysis.cg.ds.CallSite;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Stream;

/*
    A Utility class to read/write CGs to the disk.
 */
public class IOUtils {
    public static Set<CallSite> convertCGToEdges(final CallGraph cg)  {
        final var resolveInterfaces = true;
        var resolvedCallSites = new HashSet<CallSite>();
        for (Iterator<CGNode> it = cg.iterator(); it.hasNext();) {
            CGNode cgnode = it.next();
            IMethod m1 = cgnode.getMethod();
            TypeName t1 = m1.getDeclaringClass().getName();
            Selector sel1 = m1.getSelector();
            String name1 = sel1.getName().toString();
            var nodeUri = CGUtils.formatMethod(t1, name1, sel1);
            if (nodeUri == null) {
                continue;
            }
            var sourceNode = new CallGraphNode(nodeUri, CGUtils.extractMethodLineNumber(cgnode),
                    m1.getDeclaringClass().getClassLoader().getReference());

            //Record if this is a fakeRoot/boot method or not
            boolean bootSrcMethod = (sourceNode.nodeURI.equals("com/ibm/wala/FakeRootClass.fakeRootMethod:()V")
                    || sourceNode.nodeURI.equals("com/ibm/wala/FakeRootClass.fakeWorldClinit:()V"));
            if (bootSrcMethod) {
                sourceNode.nodeURI = "<boot>"; // A bootstrap method
            }

            for(Iterator<CallSiteReference> it2 = cgnode.iterateCallSites(); it2.hasNext(); ) {
                CallSiteReference csRef = it2.next();
                CallGraphNode targetNode;
                var resolvedTargets = new ArrayList<String>();
                /* Choose to resolve the interface edge or not based on the input */
                if (resolveInterfaces){
                    Set<CGNode> possibleActualTargets = cg.getPossibleTargets(cgnode, csRef);
                    for (CGNode cgnode2 : possibleActualTargets){
                        IMethod m2 = cgnode2.getMethod();
                        TypeName t2 = m2.getDeclaringClass().getName();
                        Selector sel2 = m2.getSelector();
                        String name2 = sel2.getName().toString();
                        var targetNodeUri = CGUtils.formatMethod(t2, name2, sel2);
                        if (targetNodeUri == null){
                            continue;
                        }
                        targetNode = new CallGraphNode(targetNodeUri, CGUtils.extractMethodLineNumber(cgnode2),
                                m2.getDeclaringClass().getClassLoader().getReference());
                        if (targetNode.nodeURI.equals("com/ibm/wala/FakeRootClass.fakeWorldClinit:()V")) {
                            continue;
                        }
                        if (!(sourceNode.nodeClassLoaderRef.equals(ClassLoaderReference.Primordial) &&
                                targetNode.nodeClassLoaderRef.equals(ClassLoaderReference.Primordial))) {
                            resolvedTargets.add(targetNode.nodeURI);
                        }


//                        String formattedOutputLine =  CGUtils.formatFinalOutput(sourceNode, targetNode, bootSrcMethod,
//                                csref.getProgramCounter());
//                        if (formattedOutputLine!=null){
//                            fw.write(formattedOutputLine);
//                        }
                    }
                } else {
                    MethodReference m2 = csRef.getDeclaredTarget();
                    TypeName t2 = m2.getDeclaringClass().getName();
                    Selector sel2 = m2.getSelector();
                    String name2 = sel2.getName().toString();
                    var targetNodeUri = CGUtils.formatMethod(t2, name2, sel2);
                    if (targetNodeUri == null) {
                        continue;
                    }
                    targetNode = new CallGraphNode(targetNodeUri, m2.getDeclaringClass().getClassLoader());
                    if (targetNode.nodeURI.equals("com/ibm/wala/FakeRootClass.fakeWorldClinit:()V")) {
                        continue;
                    }
                    if (!(sourceNode.nodeClassLoaderRef.equals(ClassLoaderReference.Primordial) &&
                            targetNode.nodeClassLoaderRef.equals(ClassLoaderReference.Primordial))) {
                        resolvedTargets.add(targetNode.nodeURI);
                    }
//                    String formattedOutputLine = CGUtils.formatFinalOutput(sourceNode, targetNode, bootSrcMethod,
//                            csref.getProgramCounter());
//                    if (formattedOutputLine!=null){
//                        fw.write(formattedOutputLine);
//                    }
                }
//                    if (targetNode != null) {
//                        if (!(sourceNode.nodeClassLoaderRef.equals(ClassLoaderReference.Primordial) &&
//                                targetNode.nodeClassLoaderRef.equals(ClassLoaderReference.Primordial))) {
//                        var resolvedEdge = CGUtils.formatFinalOutput(sourceNode, targetNode, bootSrcMethod);
//                        if (resolvedEdge != null) {
//                            resolvedEdges.add(resolvedEdge);
//                        }
//                    }
//                }
                if (resolvedTargets.size() != 0) {
                    resolvedCallSites.add(new CallSite(sourceNode.nodeURI, resolvedTargets));
                }
            }
        }
        return resolvedCallSites;
    }
    public static void writeWalaCGToFile(final CallGraph cg, final String programName) throws IOException {
        var file = Paths.get(programName + ".csv").toFile();
        var fw = new FileWriter(file);
        fw.write("source,offset,target,source_ln,target_ln\n"); //Header line
        CGUtils.writeCGToFile(cg, fw);
        fw.close();
    }

    public static <K, V> Stream<Map.Entry<K, V>> makeParallelStream(Map<K, V> map) {
        Stream<Map.Entry<K, V>> stream = GlobalConstants.useParallelism
                ? map.entrySet().parallelStream()
                : map.entrySet().stream();
        return stream;
    }

    public static List<File> findJarsInDirectory(String libFolder) {
        File directory = new File(libFolder);
        List<File> jarFiles = new ArrayList<>();

        if (directory.exists() && directory.isDirectory()) {
            File[] filesInDirectory = directory.listFiles();
            for (File file : filesInDirectory) {
                if (file.isFile() && file.getName().endsWith(".jar")) {
                    jarFiles.add(file);
                }
            }
        }
        return jarFiles;
    }
}
