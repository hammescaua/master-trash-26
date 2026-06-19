/*
  mastertrash_arduino.ino

  Controla o sensor HC-SR04 e o servo MG995.
  Detecta objetos e os comunica ao Raspberry Pi via Serial.
  Recebe comandos e posiciona o servo na calha correta.

  Posições do servo:
    paper   ->  45°
    plastic ->  90° (neutro)
    metal   -> 135°
    organic -> 180°
*/

#include <Servo.h>

// --- Pinos ---
const int TRIG_PIN    = 9;
const int ECHO_PIN    = 10;
const int SERVO_PIN   = 6;

// --- Configurações ---
const int DISTANCE_THRESHOLD_CM = 15;   // Distância mínima para detectar objeto
const int SERVO_NEUTRAL          = 90;  // Posição neutra (aguardando)
const int SERVO_WAIT_MS          = 3000; // Tempo com servo na posição (ms)

// --- Ângulos por classe ---
const int ANGLE_PAPER   = 45;
const int ANGLE_PLASTIC = 90;
const int ANGLE_METAL   = 135;
const int ANGLE_ORGANIC = 180;

Servo servo;
bool waitingForCommand = false; // Evita enviar OBJECT_DETECTED repetidamente

// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(9600);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  servo.attach(SERVO_PIN);
  servo.write(SERVO_NEUTRAL);

  delay(500);
  Serial.println("MASTERTRASH_READY");
}

// ---------------------------------------------------------------------------
void loop() {
  if (!waitingForCommand) {
    checkSensor();
  }

  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    processCommand(command);
  }
}

// ---------------------------------------------------------------------------
void checkSensor() {
  int distance = measureDistance();
  if (distance > 0 && distance <= DISTANCE_THRESHOLD_CM) {
    Serial.println("OBJECT_DETECTED");
    waitingForCommand = true;
  }
}

// ---------------------------------------------------------------------------
int measureDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000); // Timeout 30ms
  if (duration == 0) return -1;                   // Sem leitura válida

  return (int)(duration * 0.034 / 2);             // Converte para cm
}

// ---------------------------------------------------------------------------
void processCommand(String command) {
  if (command == "MOVE_PAPER") {
    movePaper();
  } else if (command == "MOVE_PLASTIC") {
    movePlastic();
  } else if (command == "MOVE_METAL") {
    moveMetal();
  } else if (command == "MOVE_ORGANIC") {
    moveOrganic();
  }
  // Comandos desconhecidos são ignorados silenciosamente
}

// ---------------------------------------------------------------------------
void movePaper() {
  servo.write(ANGLE_PAPER);
  delay(SERVO_WAIT_MS);
  returnNeutral();
}

void movePlastic() {
  servo.write(ANGLE_PLASTIC);
  delay(SERVO_WAIT_MS);
  returnNeutral();
}

void moveMetal() {
  servo.write(ANGLE_METAL);
  delay(SERVO_WAIT_MS);
  returnNeutral();
}

void moveOrganic() {
  servo.write(ANGLE_ORGANIC);
  delay(SERVO_WAIT_MS);
  returnNeutral();
}

void returnNeutral() {
  servo.write(SERVO_NEUTRAL);
  waitingForCommand = false; // Pronto para detectar novo objeto
}
