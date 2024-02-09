package dev.c0pslab.analysis.cg;

import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.Entrypoint;
import com.ibm.wala.ipa.callgraph.impl.DefaultEntrypoint;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.types.ClassLoaderReference;

import java.util.ArrayList;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

public class EntryPointsGenerator {
        private final IClassHierarchy cha;

        public EntryPointsGenerator(final IClassHierarchy cha) {
            this.cha = cha;
        }

        public ArrayList<Entrypoint> getEntryPoints() {
            return StreamSupport.stream(cha.spliterator(), false)
                    .filter(EntryPointsGenerator::isPublicClass)
                    .flatMap(klass -> klass.getDeclaredMethods().parallelStream())
                    .filter(EntryPointsGenerator::isPublicMethod)
                    .map(m -> new DefaultEntrypoint(m, cha))
                    .collect(Collectors.toCollection(ArrayList::new));
        }

        private static boolean isPublicClass(final IClass klass) {
            return isApplicationORLibrary(klass)
                    && !klass.isInterface()
                    && klass.isPublic();
        }

        private static boolean isPublicMethod(final IMethod method) {
            return isApplicationORLibrary(method.getDeclaringClass())
                    && method.isPublic()
                    && !method.isAbstract();
        }

        private static boolean isPublicOrPrivateMethod(final IMethod method) {
            return isApplicationORLibrary(method.getDeclaringClass())
                    && (method.isPublic() || method.isPrivate())
                    && !method.isAbstract();
        }

        private static Boolean isApplication(final IClass klass) {
            return klass.getClassLoader().getReference().equals(ClassLoaderReference.Application);
        }

        private static Boolean isApplicationORLibrary(final IClass klass) {
            return klass.getClassLoader().getReference().equals(ClassLoaderReference.Application) ||
                    klass.getClassLoader().getReference().equals(ClassLoaderReference.Extension);
        }

        private static Boolean isPrimordial(final IClass klass) {
            return klass.getClassLoader().getReference().equals(ClassLoaderReference.Primordial);
        }

        private static Boolean considerAllClasses(final IClass klass) {
            return klass.getClassLoader().getReference().equals(ClassLoaderReference.Application) ||
                    klass.getClassLoader().getReference().equals(ClassLoaderReference.Extension) ||
                    klass.getClassLoader().getReference().equals(ClassLoaderReference.Primordial);
        }
}
