package dev.c0pslab.analysis.cg;

import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.classLoader.ShrikeBTMethod;
import com.ibm.wala.core.util.strings.StringStuff;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.types.Descriptor;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.Selector;
import com.ibm.wala.types.TypeName;
import dev.c0pslab.analysis.cg.ds.CallGraphNode;
import dev.c0pslab.analysis.cg.ds.Edge;
import dev.c0pslab.analysis.utils.IOUtils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class CGUtils {
    final private static String lambdaMetafactoryClass = "java/lang/invoke/LambdaMetafactory";
    final private static String walaArrayCopy = "com/ibm/wala/model/java/lang/System.arraycopy:(Ljava/lang/Object;Ljava/lang/Object;)V";
    final private static String walaLambdaStartString = "wala/lambda$";
    final private static String javaLibArrayCopy = "java/lang/System.arraycopy:(Ljava/lang/Object;ILjava/lang/Object;II)V";
    final static String resolveInterfaces = "true";

    public static String getMethodUri(final String className, final String methodName, final String methodSignature) {
        var methodSelector = Selector.make(methodName+methodSignature);
        var classTypeName = TypeName.findOrCreate(className.replace(";", ""));
        var methodUri = formatMethod(classTypeName, methodName, methodSelector);
        return methodUri;
    }

    /**
     * Creates a URI for methods that is compatible with WALA CGs
     */
    public static String getMethodUri(final ShrikeBTMethod shrikeMethod) {
        var methodDeclaringClassName = shrikeMethod.getDeclaringClass().getName();
        var methodSelector = shrikeMethod.getSelector();
        var methodUri = formatMethod(methodDeclaringClassName, methodSelector.getName().toString(),
                methodSelector);
        return methodUri;
    }

    public static String formatMethod(final TypeName t, final String methodName, final Selector sel){
        String qualifiedClassName = "" + ( t.getPackage() == null ? "" : t.getPackage() + "/" ) + t.getClassName();
        if (qualifiedClassName.equals(lambdaMetafactoryClass)){
            return null; //don't want to use lambda metafactory nodes. They go nowhere, and don't appear in javaq
        }
        String formattedMethod = qualifiedClassName + "." + methodName + ":" + sel.getDescriptor();
        //Modify the method if it is a lambda
        formattedMethod = reformatIfLambda(formattedMethod);
        //If it is wala arrayCopy, replace with java Arraycopy
        if (formattedMethod.equals(walaArrayCopy)){
            formattedMethod = javaLibArrayCopy;
        }
        return formattedMethod;
    }

    public static String reformatIfLambda(final String inputMethod){
        String outputMethod;
        if (inputMethod.startsWith(walaLambdaStartString)){
            String fullLambdaSignature = inputMethod.substring(walaLambdaStartString.length()); //remove wala start string
            String lambdaSignatureWithoutArgs = fullLambdaSignature.split(":")[0];

            String classname = lambdaSignatureWithoutArgs.split("\\.")[0];
            String [] classnameListFormat = classname.split("\\$");
            String lambdaIndex = classnameListFormat[classnameListFormat.length-1];
            //remove the last element (the lambda index) from the classname
            classnameListFormat = Arrays.copyOf(classnameListFormat, classnameListFormat.length-1);
            String classnameFormatted = String.join("/",classnameListFormat);

            String methodname = lambdaSignatureWithoutArgs.split("\\.")[1];
            outputMethod = classnameFormatted + ".<lambda/" + methodname + "$" + lambdaIndex + ">:()V";
            return outputMethod;
        }
        else{ //If it is not a lambda method
            return inputMethod;
        }
    }

    public static String formatFinalOutput(final CallGraphNode sourceNode, final CallGraphNode targetNode,
                                           final boolean bootSrcMethod, final int off){
        //Decide the bytecode offset (and fix firstMethod) depending on if it is a boot method
        int bytecodeOffset;
        if (bootSrcMethod){
            sourceNode.nodeURI = "<boot>";
            bytecodeOffset = 0;
        } else {
            bytecodeOffset = off;
        }

        //Skip this edge if destination node is a boot method
        if (targetNode.equals("com/ibm/wala/FakeRootClass.fakeWorldClinit:()V")){
            return null;
        }
        return sourceNode.nodeURI + "," + bytecodeOffset + "," + targetNode.nodeURI + "," +
                sourceNode.nodeLineNumbers + "," + targetNode.nodeLineNumbers + "\n";
    }

    public static Edge formatFinalOutput(final CallGraphNode sourceNode, final CallGraphNode targetNode,
                                         final boolean bootSrcMethod){
        if (bootSrcMethod){
            sourceNode.nodeURI = "<boot>";
        }
        //Skip this edge if destination node is a boot method
        if (targetNode.nodeURI.equals("com/ibm/wala/FakeRootClass.fakeWorldClinit:()V")){
            return null;
        }
        return new Edge(sourceNode.nodeURI, targetNode.nodeURI);
    }

    public static String extractMethodLineNumber(final CGNode cgNode) {
        int methodBeginLine = 0;
        int methodEndLine = 0;
        IR cgNodeIR = cgNode.getIR();
        if (cgNodeIR != null) {
            IMethod method = cgNodeIR.getMethod();
            // TODO: Might be a better workaround!
//            for (var inst : cgNodeIR.getInstructions()) {
//                if (inst.iIndex() != -1) {
//                    int currMethodLine = method.getLineNumber(inst.iIndex());
//                    if (methodBeginLine + methodEndLine == 0) {
//                        methodBeginLine = currMethodLine;
//                        methodEndLine = currMethodLine;
//                    } else if (currMethodLine > methodEndLine) {
//                        methodEndLine = currMethodLine;
//                    }
//                }
//            }

            for (Iterator<SSAInstruction> it2 = cgNodeIR.iterateAllInstructions(); it2.hasNext();) {
                SSAInstruction s = it2.next();
                if (s.iIndex() != -1) {
                    try {
                        int currMethodLine = method.getLineNumber(s.iIndex());
                        if (methodBeginLine + methodEndLine == 0) {
                            methodBeginLine = currMethodLine;
                            methodEndLine = currMethodLine;
                        } else if (currMethodLine > methodEndLine) {
                            methodEndLine = currMethodLine;
                        }
                    } catch (ArrayIndexOutOfBoundsException e) {
                        break;
                    }
                }
            }
        }
        return String.valueOf(methodBeginLine) + " " + String.valueOf(methodEndLine);
    }

    public static String getMethodSignature(final String methodUri) {
        return methodUri.substring(methodUri.indexOf(':') + 1);
    }

    public static String getClassTypeFromUri(final String uri) {
        return uri.split("\\.")[0];
    }

    public static String getMethodNameFromUri(final String uri) {
        final var pattern = Pattern.compile(".+\\.(.+):.+");
        final var matcher = pattern.matcher(uri);

        if (matcher.find()) {
            return matcher.group(1);
        } else {
            return null;  // or throw an appropriate exception if preferred
        }
    }

    public static Selector makeWALASelector(final String methodName, final String methodSignature) {
        return Selector.make(methodName + methodSignature);
    }

    public static List<File> getProgramDependencies(final Path programJarFile) {
        final var programJarFileDeps = programJarFile.getParent().resolve("dependencies").toString();
        return IOUtils.findJarsInDirectory(programJarFileDeps);
    }

    public static List<File> getProgramDependencies(final Path programJarFile, final String excludedJarFile) {
        final var programJarFileDeps = programJarFile.getParent().resolve("dependencies").toString();
        var foundProgramJarFiles = IOUtils.findJarsInDirectory(programJarFileDeps);
        return foundProgramJarFiles.stream().filter(file -> !file.getName().equals(excludedJarFile)).collect(Collectors.toList());
    }

    public static String getPackageUri(final String uri) {
        var lastSlasIndex = uri.split("\\.")[0].lastIndexOf('/');
        return uri.substring(0, lastSlasIndex + 1);
    }

    public static boolean areTwoPackageUrisSimilar(final String fistUri, final String secondUri) {
        String[] parts1 = fistUri.split("/");
        String[] parts2 = secondUri.split("/");

        List<String> commonParts = new ArrayList<>();
        final int minLength = Math.min(parts1.length, parts2.length);
        for (int i = 0; i < minLength; i++) {
            if (parts1[i].equals(parts2[i])) {
                commonParts.add(parts1[i]);
            } else {
                break;  // Exit loop once a non-common part is found
            }
        }
        if (GlobalConstants.minCommonPackageUriParts < minLength) {
            return commonParts.size() >= GlobalConstants.minCommonPackageUriParts;
        } else {
            return commonParts.size() >= GlobalConstants.minCommonPackageUriParts - minLength;
        }
    }

    public static Path createWalaExclusionFile() throws IOException {
        final List<String> exclusions = Arrays.asList(
//                "java/.*",
                "java/util/.*",
                "java/io/.*",
                "java/nio/.*",
                "java/net/.*",
                "java/math/.*",
                "java/awt/.*",
                "java/text/.*",
                "java/sql/.*",
                "java/security/.*",
                "java/time/.*",
                "javax/.*",
                "sun/.*",
                "com/sun/.*",
                "jdk/.*",
                "org/graalvm/.*"
        );

        final var tmpFilePath = Paths.get("/tmp/wala_exclusions.txt");
        Files.write(tmpFilePath, exclusions);
        return tmpFilePath;
    }

    public static String simplifyMethodSignature(String methodSignature) {
        Descriptor descriptor = Descriptor.findOrCreateUTF8(methodSignature);
        StringBuilder methodSig = new StringBuilder("(");
        final TypeName[] methodParams = descriptor.getParameters();
        if (methodParams != null) {
            for (TypeName p: methodParams) {
                if (methodSig.toString().equals("(")) {
                    methodSig.append(StringStuff.dollarToDot(p.getClassName().toString()));
                } else {
                    methodSig.append(", ").append(StringStuff.dollarToDot(p.getClassName().toString()));
                }
            }
        }
        return methodSig + ")" + descriptor.getReturnType().getClassName();
    }

    public static void writeCGToFile(CallGraph cg, FileWriter fw) throws IOException {
        for(Iterator<CGNode> it = cg.iterator(); it.hasNext();) {
            CGNode cgnode = it.next();
            IMethod m1 = cgnode.getMethod();
            TypeName t1 = m1.getDeclaringClass().getName();
            Selector sel1 = m1.getSelector();
            String name1 = sel1.getName().toString();
            CallGraphNode sourceNode = new CallGraphNode();
            sourceNode.nodeURI = formatMethod(t1,name1,sel1);

            if (sourceNode.nodeURI == null) {
                continue;
            }

            sourceNode.nodeLineNumbers = extractMethodLineNumber(cgnode);

            //Record if this is a fakeRoot/boot method or not
            boolean bootSrcMethod = (sourceNode.nodeURI.equals("com/ibm/wala/FakeRootClass.fakeRootMethod:()V")
                    || sourceNode.nodeURI.equals("com/ibm/wala/FakeRootClass.fakeWorldClinit:()V"));

            for(Iterator<CallSiteReference> it2 = cgnode.iterateCallSites(); it2.hasNext(); ) {
                CallSiteReference csref = it2.next();
                /* Choose to resolve the interface edge or not based on the input */
                if (resolveInterfaces.equalsIgnoreCase("true")){
                    Set<CGNode> possibleActualTargets = cg.getPossibleTargets(cgnode, csref);
                    for (CGNode cgnode2 : possibleActualTargets){
                        IMethod m2 = cgnode2.getMethod();
                        TypeName t2 = m2.getDeclaringClass().getName();
                        Selector sel2 = m2.getSelector();
                        String name2 = sel2.getName().toString();
                        CallGraphNode targetNode = new CallGraphNode();
                        targetNode.nodeURI = formatMethod(t2,name2,sel2);

                        if (targetNode.nodeURI == null){
                            continue;
                        }

                        targetNode.nodeLineNumbers = extractMethodLineNumber(cgnode2);

                        String formattedOutputLine =  formatFinalOutput(sourceNode, targetNode,bootSrcMethod,csref.getProgramCounter());
                        if (formattedOutputLine!=null){
                            fw.write(formattedOutputLine);
                        }
                    }
                } else {
                    MethodReference m2 = csref.getDeclaredTarget();
                    TypeName t2 = m2.getDeclaringClass().getName();
                    Selector sel2 = m2.getSelector();
                    String name2 = sel2.getName().toString();
                    CallGraphNode targetNode = new CallGraphNode();
                    targetNode.nodeURI = formatMethod(t2,name2,sel2);

                    if (targetNode.nodeURI == null) {
                        continue;
                    }

                    String formattedOutputLine = formatFinalOutput(sourceNode, targetNode,bootSrcMethod,csref.getProgramCounter());
                    if (formattedOutputLine!=null){
                        fw.write(formattedOutputLine);
                    }
                }
            }
        }
    }
}
