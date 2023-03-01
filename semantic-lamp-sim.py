from functions.semanticlamp import SemanticLamp
import pygame
import time
# MAIN PROGRAM


print("Starting...")

# load model
semanticlamp = SemanticLamp(pygamesim=True)

print("Begin.")
# record, decode, process, repeat
RUNNING = True
while RUNNING:
    for event in pygame.event.get():
        pygame.event.wait()
        print('What do you have to say?')
        captured_text = semanticlamp.capture()

        if captured_text:
            if 'quit' in str(captured_text):
                print('OK, bye.')
                pygame.quit()
                RUNNING = False
                break
            else:
                print(f"Heard: {captured_text}")
                start_at = time.time()
                label, score = semanticlamp.predict(captured_text)
                print(
                    f"Predicted sentiment: {label}, Score: {float(score)} , elapsed_time: {time.time()-start_at}")

                # update colour
                semanticlamp.update_colour(label, score)

        else:
            continue
