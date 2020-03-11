#!/bin/sh
function usage {
    echo "usage: $0 "
    echo "  -i      subid (mandatory)"
    echo "  -o      order (integer: 1-4, mandatory)"
    echo "  -t      task"
    exit 1
}

while getopts hi:o:t: option
do
        case "${option}"
        in
        		h) usage;;
                i) subid=${OPTARG};;
                o) scanner_order=${OPTARG};;
                t) task=${OPTARG};;
        esac
done

echo "Creating battery for subject: $subid"
echo "Task: $task"
expfactory --run --folder scanner_tasks_order$scanner_order/ --battery expfactory-battery/ --experiments $task --subid $subid