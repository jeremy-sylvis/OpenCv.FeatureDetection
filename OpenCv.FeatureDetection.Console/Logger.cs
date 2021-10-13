using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpenCv.FeatureDetection.Console
{
    /// <summary>
    /// A quick proxy for logging to console. This serves as a nice separation point for other logging implementations, should we need them.
    /// </summary>
    public class Logger
    {
        public void WriteMessage(string message)
        {
            System.Console.WriteLine(message);
        }
    }
}
