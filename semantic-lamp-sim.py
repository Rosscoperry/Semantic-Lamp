# MAIN PROGRAM


print("Starting...")

# load model


# initalize pygame

# init pygame
# Initializing Pygame


print("GO!")

# record, decode, process, repeat
while 1:
    print('What do you have to say?')
    captured_text = capture()

    if captured_text == 0:
        continue

    if 'quit' in str(captured_text):
        print('OK, bye.')
        pygame.quit()
        break

    # Process captured text
    process_text(captured_text)
