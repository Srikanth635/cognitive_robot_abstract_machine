using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;

public static class MemoryJob
{
    [StructLayout(LayoutKind.Sequential)]
    private struct BasicLimits
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct Counters
    {
        public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
        public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct ExtendedLimits
    {
        public BasicLimits BasicLimitInformation;
        public Counters IoInfo;
        public UIntPtr ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed;
    }

    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    private static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError=true)]
    private static extern bool SetInformationJobObject(IntPtr job, int information, ref ExtendedLimits limits, uint length);
    [DllImport("kernel32.dll", SetLastError=true)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError=true)]
    private static extern bool QueryInformationJobObject(IntPtr job, int information, out ExtendedLimits limits, uint length, out uint returnedLength);
    [DllImport("kernel32.dll")]
    private static extern bool CloseHandle(IntPtr handle);

    public static int Run(string executable, string worker, string directory, string[] arguments, ulong maximumBytes, string statisticsPath = null)
    {
        IntPtr job = CreateJobObject(IntPtr.Zero, null);
        if (job == IntPtr.Zero) throw new Win32Exception();
        var limits = new ExtendedLimits();
        limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x100 | 0x200;
        limits.ProcessMemoryLimit = new UIntPtr(maximumBytes);
        limits.JobMemoryLimit = new UIntPtr(maximumBytes);
        if (!SetInformationJobObject(job, 9, ref limits, (uint)Marshal.SizeOf(limits))) throw new Win32Exception();
        var start = new ProcessStartInfo(executable);
        start.WorkingDirectory = directory;
        start.UseShellExecute = false;
        start.CreateNoWindow = true;
        start.RedirectStandardInput = true;
        start.ArgumentList.Add(worker);
        foreach (string argument in arguments) start.ArgumentList.Add(argument);
        var process = Process.Start(start);
        if (process == null) throw new InvalidOperationException("No worker was started.");
        if (!AssignProcessToJobObject(job, process.Handle))
        {
            process.Kill(true);
            throw new Win32Exception();
        }
        process.StandardInput.WriteLine("run");
        process.StandardInput.Close();
        process.WaitForExit();
        int result = process.ExitCode;
        if (!string.IsNullOrEmpty(statisticsPath))
        {
            ExtendedLimits measured;
            uint returnedLength;
            if (!QueryInformationJobObject(job, 9, out measured, (uint)Marshal.SizeOf(limits), out returnedLength)) throw new Win32Exception();
            File.WriteAllText(statisticsPath,
                "exit_status,peak_job_committed_bytes,peak_process_committed_bytes,maximum_bytes\n" +
                result.ToString(CultureInfo.InvariantCulture) + "," +
                measured.PeakJobMemoryUsed.ToUInt64().ToString(CultureInfo.InvariantCulture) + "," +
                measured.PeakProcessMemoryUsed.ToUInt64().ToString(CultureInfo.InvariantCulture) + "," +
                maximumBytes.ToString(CultureInfo.InvariantCulture) + "\n");
        }
        process.Dispose();
        CloseHandle(job);
        return result;
    }
}

