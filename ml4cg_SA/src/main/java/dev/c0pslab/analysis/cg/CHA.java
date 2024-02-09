package dev.c0pslab.analysis.cg;

import com.ibm.wala.core.util.config.AnalysisScopeReader;
import com.ibm.wala.ipa.callgraph.AnalysisScope;
import com.ibm.wala.ipa.cha.ClassHierarchy;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ipa.cha.ClassHierarchyFactory;
import com.ibm.wala.types.ClassLoaderReference;
import com.ibm.wala.util.config.FileOfClasses;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.List;
import java.util.jar.JarFile;

public class CHA {
    private static final Logger LOG = LoggerFactory.getLogger(CHA.class);
    private AnalysisScope scope;
    private ClassHierarchy cha;
    public ClassHierarchy construct(final String programJarFile, final List<File> dependenciesJarFiles) {
        try {
            scope = AnalysisScopeReader.instance.makeJavaBinaryAnalysisScope(programJarFile, null);
            for (var depJarFile: dependenciesJarFiles) {
                scope.addToScope(ClassLoaderReference.Extension, new JarFile(depJarFile));
            }
            cha = ClassHierarchyFactory.make(scope);
            LOG.info("Built CHA with " + dependenciesJarFiles.size() + " dependencies");
        } catch (IOException | ClassHierarchyException e) {
            throw new RuntimeException(e);
        }
        return cha;
    }

    public ClassHierarchy construct(final String programJarFile, final List<File> dependenciesJarFiles,
                                    final File exclusionFile) {
        try {
            scope = AnalysisScopeReader.instance.makeJavaBinaryAnalysisScope(programJarFile, null);
            for (var depJarFile: dependenciesJarFiles) {
                scope.addToScope(ClassLoaderReference.Extension, new JarFile(depJarFile));
            }
            try (var fis = new FileInputStream(exclusionFile)) {
                scope.setExclusions(new FileOfClasses(fis));
            }

            cha = ClassHierarchyFactory.make(scope);
            LOG.info("Built CHA with " + dependenciesJarFiles.size() + " dependencies");
        } catch (IOException | ClassHierarchyException e) {
            throw new RuntimeException(e);
        }
        return cha;
    }

    public AnalysisScope getCHAScope() {
        return scope;
    }
}
