#include <cstdio>
#include <cassert>
#include <cstring>

#include <iostream>
#include <fstream>

#include "fluid.hpp"

using namespace std;

#define error_exit(fmt, ...) do { fprintf(stderr, "%s error: " fmt, __func__, ##__VA_ARGS__); exit(1); } while(0);

Fluid fluid;
int NCORES = 1;
int INDEX = 0;
int ITERATION = 0;
std::ofstream myfile;

bool start = false;

void displayCallback( void ) {
    struct timespec before, after;

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.2, 0.3, 0.6, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    if (start) {
        ITERATION++;
        clock_gettime(CLOCK_REALTIME, &before);
        
        switch (INDEX) {
            case 0:
                fluid.simulate_seq();
                break;
            case 1: 
                fluid.simulate_omp(NCORES);
                break;
            case 2: 
                fluid.simulate_cuda();
                break;
        }

        clock_gettime(CLOCK_REALTIME, &after);

        double delta_ms = (double) (after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;
        myfile << delta_ms << "\n";
        std::cout << ITERATION << std::endl;
    }

    fluid.draw();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(110.0, 1024.0 / 768.0, 0.05, 100.0);

    glutSwapBuffers();
}

void keyboardCallback( unsigned char key, int x, int y ) {
    if (key == 's') // Start to simulate
        start = true;

    if (key == 'q') {
        myfile.close();
        exit(0);
    }
}

void setLighting( void ) {
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    float lightPos[] = {0.0f, 0.4f, 1.0f, 0.0f};
    float lightAmb[] = {0.0f, 0.0f, 0.0f, 1.0f};
    float lightDif[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float lightSpc[] = {1.0f, 1.0f, 1.0f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
    glLightfv(GL_LIGHT0, GL_AMBIENT , lightAmb);
    glLightfv(GL_LIGHT0, GL_DIFFUSE , lightDif);
    glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpc);

    float matAmb[] = {0.7f, 0.7f, 0.9f, 1.0f};
    float matDif[] = {0.7f, 0.7f, 0.9f, 1.0f};
    float matSpc[] = {0.0f, 0.0f, 0.0f, 1.0f};
    float matShi[] = {1.0f, 1.0f, 1.0f, 1.0f};
    glMaterialfv(GL_FRONT, GL_AMBIENT  , matAmb);
    glMaterialfv(GL_FRONT, GL_DIFFUSE  , matDif);
    glMaterialfv(GL_FRONT, GL_SPECULAR , matSpc);
    glMaterialfv(GL_FRONT, GL_SHININESS, matShi);
}

void display() {
    glClearColor(1.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    glFlush();
}

int main( int argc, char *argv[] ) {
    if (argc < 3 || strcmp(argv[1], "-v") != 0) {
        error_exit("Expecting argument: -v [process type]\n");
    }

    INDEX = atoi(argv[2]);
    if (INDEX > 2 || INDEX < 0) {
        error_exit("Illegal process type: %d\n", INDEX);
    }

    if (argc == 5 && strcmp(argv[3], "-p") == 0) {
        NCORES = atoi(argv[4]);
    }

    if (INDEX == 2 && NCORES < 1) {
        error_exit("Illegal core count: %d\n", NCORES);
    } else if (INDEX == 1 && NCORES < 1) {
        error_exit("Illegal core count: %d\n", NCORES);
    }

    myfile.open ("cuda.csv");

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(1024, 768);
    glutCreateWindow("SPH Animation");

    glutDisplayFunc(displayCallback);
    glutIdleFunc(displayCallback);
    glutKeyboardFunc(keyboardCallback);

    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);
    setLighting();
    glutMainLoop();

    return 0;
}
