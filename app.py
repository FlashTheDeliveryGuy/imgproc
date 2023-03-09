from templatescanner import Template_Page
from multiprocessing import Lock, Process, Queue, current_process
import fitz

#print("Number of cpu : ", multiprocessing.cpu_count())



#queue = multiprocessing.Queue()

#for page in doc:

    #print('scanning and cropping page ')

    #queue.put(Template_Page(page))
    #print(f'Queuing page {page}')
    #crop = temp.crop()

    #plt.imshow(crop)
    #plt.show()

def main():

    doc = fitz.open('testdocuments/scantest2.pdf')

    number_of_task = 10
    number_of_processes = 4
    pages_to_process = Queue()
    crops_done = Queue()
    processes = []

    for page in doc:
        current = Template_Page(page)
        pages_to_process.put(current)
        print('putting page')
        print(current.getID())

    # creating processes
    for w in range(number_of_processes):
        p = Process(target=pages_to_process.crop())
        processes.append(p)
        p.start()

    # completing process
    for p in processes:
        p.join()

    # print the output
    while not tasks_that_are_done.empty():
        print(tasks_that_are_done.get())

    return True

if __name__ == '__main__':
    main()